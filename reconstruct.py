import os
import json
import torch
import argparse
import random

from ptb import *
from model import SentenceVAE
from utils import to_var, idx2word, interpolate, similarity

def main(args):
    with open(f'{args.data_dir}/{args.data_prefix}.vocab.json', 'r') as file:
        vocab = json.load(file)

    if args.conditioned:
        with open(f'{args.data_dir}/{args.data_prefix}.cond_vocab.json', 'r') as file:
            cond_vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    data = PTB(
        data_dir=args.data_dir,
        create_data=False,
        split=args.split,
        max_sequence_length=args.max_sequence_length,
        data_prefix=args.data_prefix,
        conditioned=args.conditioned,
        bars=args.bars
    )

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=len(w2i),
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        conditioned=args.conditioned,
        cond_size=data.cond_vocab_size
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    cuda = torch.cuda.is_available()

    if cuda:
        model.load_state_dict(torch.load(args.load_checkpoint))
    else:
        model.load_state_dict(torch.load(args.load_checkpoint, map_location=torch.device("cpu")))

    print("Model loaded from %s" % args.load_checkpoint)

    if cuda:
        model = model.cuda()

    model.eval()

    num = random.randint(0, len(data))
    tune = torch.tensor(np.array([data[num]['input']]))
    length = [data[num]['length']]
    tune_str = idx2word(tune, i2w=i2w, pad_idx=w2i['<pad>'])[0]

    if args.conditioned:
        cond = torch.tensor(np.array([data[num]['conditioning']]))
        if cuda:
            cond = cond.cuda()
    else:
        cond = None

    if cuda:
        logp, mean, logv, z, z_cond = model(tune.cuda(),
                                    torch.tensor(length).cuda(),
                                    cond)
    else:
        logp, mean, logv, z, z_cond = model(tune,
                                    torch.tensor(length),
                                    cond)


    samples, z = model.inference(n=1, z=z_cond, mode=args.mode,
                                 T=args.temperature, P=args.topp, K=args.topk)

    print('----------SAMPLES----------')
    output = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])
    output.insert(0, tune_str)


    output = [o.split(' ') for o in output]
    for i, o in enumerate(output):
        if '<sos>' in o:
            o.remove('<sos>')
        if '<eos>' in o:
            o.remove('<eos>')

    output = [' '.join(o) for o in output]
    output = '\n'.join(output)

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
    parser.add_argument('-bb', '--bars', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.mode = args.mode.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.mode in ['greedy', 'topk', 'topp']
    assert args.split in ['train', 'test', 'valid']
    assert 0 <= args.word_dropout <= 1

    main(args)
