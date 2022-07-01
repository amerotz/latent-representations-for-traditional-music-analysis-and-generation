import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from utils import OrderedCounter

class PTB(Dataset):

    def __init__(self, data_dir, split, create_data, conditioned, bars, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.data_prefix = kwargs.get('data_prefix', 'data_v2_cleaned')
        self.max_sequence_length = kwargs.get('max_sequence_length', 256)

        self.all_splits_path = os.path.join(data_dir, self.data_prefix + '.txt')

        self.raw_data_path = os.path.join(data_dir, f'{self.data_prefix}.{split}.txt')
        self.data_file = f'{self.data_prefix}.{split}.json'
        self.vocab_file = f'{self.data_prefix}.vocab.json'
        self.conditioned = conditioned
        self.bars = bars;

        if self.conditioned:
            self.cond_vocab_file = f'{self.data_prefix}.cond_vocab.json'
            self.data_file = f'{self.data_prefix}.{split}.cond.json'

        if create_data:
            print("Creating new %s ptb data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        if self.conditioned:
            if self.bars:
                return {
                    'input': np.asarray(self.data[idx]['input']),
                    'target': np.asarray(self.data[idx]['target']),
                    'length': self.data[idx]['length'],
                    'key': self.data[idx]['key'],
                    'meter': self.data[idx]['meter'],
                    'function': self.data[idx]['function'],
                    'conditioning': np.asarray(self.data[idx]['conditioning'])
                }
            else:
                return {
                    'input': np.asarray(self.data[idx]['input']),
                    'target': np.asarray(self.data[idx]['target']),
                    'length': self.data[idx]['length']
                }
        else:
            if self.bars:
                return {
                    'input': np.asarray(self.data[idx]['input']),
                    'target': np.asarray(self.data[idx]['target']),
                    'length': self.data[idx]['length'],
                    'key': self.data[idx]['key'],
                    'meter': self.data[idx]['meter'],
                    'function': self.data[idx]['function']
                }
            else:
                return {
                    'input': np.asarray(self.data[idx]['input']),
                    'target': np.asarray(self.data[idx]['target']),
                    'length': self.data[idx]['length'],
                }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def cond_vocab_size(self):
        if self.conditioned:
            return len(self.cond_vocab)
        else:
            return 0

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:

            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)

            if self.conditioned:
                with open(os.path.join(self.data_dir, self.cond_vocab_file), 'r') as file:
                    self.cond_vocab = json.load(file)

            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        if self.conditioned:
            with open(os.path.join(self.data_dir, self.cond_vocab_file), 'r') as vocab_file:
                self.cond_vocab = json.load(vocab_file)


        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()


        data = defaultdict(dict)


        with open(self.raw_data_path, 'r') as file:

            for i, line in enumerate(file):

                if self.bars:
                    raw = line.replace('\n', '')
                    raw = raw.split('\t')
                    words = raw[0]
                else:
                    words = line.replace('\n', '')

                if self.conditioned:
                    cond = raw[1:]
                    conditioning = [0 for _ in range(len(self.cond_vocab))]
                    for c in cond:
                        conditioning[self.cond_vocab[c]] = 1

                words = words.split(' ')

                input = ['<sos>'] + words
                input = input[:self.max_sequence_length]

                target = words[:self.max_sequence_length-1]
                target = target + ['<eos>']

                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)

                input.extend(['<pad>'] * (self.max_sequence_length-length))
                target.extend(['<pad>'] * (self.max_sequence_length-length))

                input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
                target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

                if self.bars:
                    data[id]['key'] = raw[1]
                    data[id]['meter'] = raw[2]
                    data[id]['function'] = raw[3]

                if self.conditioned:
                    data[id]['conditioning'] = conditioning

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocabulary can only be created for training file."

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(self.all_splits_path, 'r') as file:
            #with open('data/data_v2_cleaned.txt', 'r') as file:

            if self.conditioned:
                self.cond_vocab = {}
                cond_vocab_index = 0

            for i, line in enumerate(file):

                if self.bars:
                    raw = line.replace('\n', '')
                    raw = raw.split('\t')
                    words = raw[0]

                else:
                    words = line.replace('\n', '')

                words = words.split(' ')
                w2c.update(words)

                if self.conditioned:
                    cond = raw[1:]
                    for c in cond:
                        if not c in self.cond_vocab:
                            self.cond_vocab[c] = cond_vocab_index
                            cond_vocab_index += 1

                    self.cond_len = len(self.cond_vocab)

            for w, c in w2c.items():
                if w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocabulary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        if self.conditioned:
            with io.open(os.path.join(self.data_dir, self.cond_vocab_file), 'wb') as vocab_file:
                data = json.dumps(self.cond_vocab, ensure_ascii=False)
                vocab_file.write(data.encode('utf8', 'replace'))


        self._load_vocab()


