import os
import re
import json
import torch
import argparse
import random

from ptb import *
from model import SentenceVAE
from utils import to_var, idx2word, interpolate

matrix = {
    'c': np.array([1, 1, 0]), # I IV VI
    'd': np.array([0, 1, 1]), # II V VII
    'e': np.array([1, 0, 0]), # I III VI
    'f': np.array([0, 1, 1]), # II IV VII
    'g': np.array([1, 0, 1]), # I III V 
    'a': np.array([1, 1, 0]), # II IV VI
    'b': np.array([1, 0, 1])  # III V VII
}

functions = ['T', 'P', 'D']

def function(x):

    f = np.zeros((3,))

    tmp = re.sub(r'[^a-gA-G]', '', x).lower()
    for t in tmp:
        f += matrix[t]

    index = np.argmax(f)

    return functions[index]


def main(args):

    torch.manual_seed(0)

    # load dicts
    with open(f'data/bars_no_key_conditioned.vocab.json', 'r') as file: bar_vocab = json.load(file)
    with open(f'data/bars_no_key_conditioned.cond_vocab.json', 'r') as file:
        cond_vocab = json.load(file)

    with open(f'data/data_v2_cleaned.vocab.json', 'r') as file:
        vocab = json.load(file)

    bar_w2i, bar_i2w = bar_vocab['w2i'], bar_vocab['i2w']
    w2i, i2w = vocab['w2i'], vocab['i2w']

    # load model
    model = SentenceVAE(
        vocab_size=len(bar_w2i),
        sos_idx=bar_w2i['<sos>'],
        eos_idx=bar_w2i['<eos>'],
        pad_idx=bar_w2i['<pad>'],
        unk_idx=bar_w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=len(bar_w2i),
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        conditioned=args.conditioned,
        cond_size=len(cond_vocab)
        )

    # load checkpoint
    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    with open(f'data/data_v2_cleaned.{args.split}.json') as f:
        data = json.load(f)

    # pick random tune
    num = random.randint(0, len(data))
    tune = torch.tensor(np.array([data[str(num)]['input']]))
    tune_str = idx2word(tune, i2w=i2w, pad_idx=w2i['<pad>'])[0]

    # save key & time for later
    time = tune_str.split(' ')[1]
    key = tune_str.split(' ')[2]

    # append to output
    tune_str = ' '.join(tune_str.split(' ')[3:])
    output = f'{time}\n{key}\n{tune_str}'

    # process bars
    if tune_str[0] != '|':
        tune_str = '| ' + tune_str

    raw_bars = tune_str.split('|')
    # remove ''
    while '' in raw_bars:
        raw_bars.remove('')
    # add | again to each bar
    raw_bars = [f'|{b}|' for b in raw_bars]

    # translate tokens to indexes
    bars = []
    for b in raw_bars:
        tmp = []
        for wd in b.split(' '):
            tmp.append(bar_w2i[wd])
        bars.append(tmp)


    rec = f'{time}\n{key}\n'
    # for each bar
    for bar, bar_str in zip(bars, raw_bars):

        # model input
        tensor = torch.tensor(np.array([bar]))

        # create conditioning
        if args.conditioned:
            cond = [key, time, function(bar_str)]
            cond = [cond_vocab[x] for x in cond]
            multihot = np.zeros((len(cond_vocab), ))
            for c in cond:
                multihot[c] = 1
            cond = torch.tensor(np.array([multihot])).to(torch.float32)
            cond = cond.cuda()
        else:
            cond = None

        # forward
        logp, mean, logv, z, z_cond = model(tensor.cuda(),
                                    torch.tensor([len(bar)]).cuda(),
                                    cond)

        # reconstruct
        samples, z = model.inference(n=1, z=z_cond,
                                     mode=args.mode, T=args.temperature,
                                     P=args.topp, K=args.topk)

        # translate indexes to words
        string = idx2word(samples, i2w=bar_i2w, pad_idx=bar_w2i['<pad>'])[0]
        string = string.replace('<eos>', '')
        rec += string

    rec = rec.replace('| |', '|')
    print()
    output += '\n' + rec

    '''

    logp, mean, logv, z = model(tune.cuda(), torch.tensor([len(tune)]).cuda())

    samples, z = model.inference(n=1, z=z, mode=args.mode,
                                 T=args.temperature, P=args.topp, K=args.topk)

    print('----------SAMPLES----------')
    output = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])
    output.insert(0, tune_str)

    '''

    '''
    output = [o.split(' ') for o in output]
    for i, o in enumerate(output):
        if '<sos>' in o:
            o.remove('<sos>')
        if '<eos>' in o:
            o.remove('<eos>')

    output = '\n\n'.join(output)


    import music21 as m21

    tune = m21.converter.parse(output, makeNotation=False)
    tune.show()
    '''

    if args.print:
        print(output)


    if args.save:
        if not 'artifacts' in os.listdir('.'):
            os.mkdir('./artifacts')

        filename = ''

        files = os.listdir('./artifacts')

        if len(files) == 0:
            filename = '0'
        else:
            files = [int(f.split('_')[0]) for f in files]
            filename = f'{max(files) + 1}'

        filename += f'_m={args.mode}'
        if args.mode == 'topp':
            filename += f'_p={args.topp}'
        if args.mode == 'topk' or args.mode == 'topp':
            filename += f'_t={args.temperature}'
        filename += f'_ls={args.latent_size}_nl={args.num_layers}'
        filename += '.abc'

        with open(f'./artifacts/{filename}', 'w') as f:
            f.write(output)
            print(f'Written to file \'./artifacts/{filename}\'.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=256)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('-pr', '--print', action='store_true')
    parser.add_argument('-m', '--mode', type=str, default='topk')
    parser.add_argument('-k', '--topk', type=int, default=10)
    parser.add_argument('-p', '--topp', type=float, default='0.9')
    parser.add_argument('-t', '--temperature', type=float, default=1.0)
    parser.add_argument('-dp', '--data_prefix', type=str, default='data_v2_cleaned')
    parser.add_argument('-sp', '--split', type=str, default='test')
    parser.add_argument('-cc', '--conditioned', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.mode = args.mode.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.mode in ['greedy', 'topk', 'topp']
    assert args.split in ['train', 'test', 'valid']
    assert 0 <= args.word_dropout <= 1

    main(args)
