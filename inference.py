import os
import json
import torch
import argparse
import numpy as np

from model import SentenceVAE
from utils import to_var, idx2word, interpolate


def main(args):

    #torch.manual_seed(0)

    with open(f'{args.data_dir}/{args.data_prefix}.vocab.json', 'r') as file:
        vocab = json.load(file)

    if args.conditioned:
        with open(f'{args.data_dir}/{args.data_prefix}.cond_vocab.json', 'r') as file:
            cond_vocab = json.load(file)
    else:
        cond_vocab = {}


    w2i, i2w = vocab['w2i'], vocab['i2w']

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
        cond_size=len(cond_vocab)
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    torch.no_grad()

    z = torch.tensor(torch.randn([args.num_samples, args.latent_size]).numpy()).cuda()
    '''
    z2 = torch.randn([args.latent_size]).numpy()

    z = to_var(
        torch.from_numpy(
            interpolate(start=z1, end=z2, steps=args.num_samples)
        ).float()
    )
    '''

    print(z.shape)

    if args.conditioned:
        key = 'K:Cmaj'
        time = 'M:4/4'
        cond = [key, time, 'T']
        cond = [cond_vocab[x] for x in cond]
        multihot = np.zeros((len(cond_vocab), ))
        for c in cond:
            multihot[c] = 1
        cond = torch.tensor(np.array([multihot])).to(torch.float32)
        cond = cond.repeat((2+args.num_samples, 1))

        z = torch.cat((z, cond.cuda()), dim=1)

    samples, z = model.inference(z=z,
                                 mode=args.mode,
                                 T=args.temperature,
                                 K=args.topk,
                                 P=args.topp)

    print('----------SAMPLES----------')
    output = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])

    output = [o.split(' ') for o in output]
    for i, o in enumerate(output):
        if '<eos>' in o:
            o.remove('<eos>')
        o.insert(0, f'X:{i}')

    output = [
        f'{o[0]}\n{o[1]}\n{o[2]}\n' +
        ''.join(o[3:]) for o in output
    ]
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

    '''
    z1 = torch.randn([args.latent_size]).numpy()
    z2 = torch.randn([args.latent_size]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    samples, _ = model.inference(z=z)
    print('-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
    '''


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
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('-pr', '--print', action='store_true')
    parser.add_argument('-m', '--mode', type=str, default='topk')
    parser.add_argument('-k', '--topk', type=int, default=10)
    parser.add_argument('-p', '--topp', type=float, default='0.9')
    parser.add_argument('-t', '--temperature', type=float, default=1.0)
    parser.add_argument('-cc', '--conditioned', action='store_true')
    parser.add_argument('-dp', '--data_prefix', type=str, default='data_v2_cleaned')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.mode = args.mode.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.mode in ['greedy', 'topk', 'topp']
    assert 0 <= args.word_dropout <= 1

    main(args)
