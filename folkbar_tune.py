import os
import json
import torch
import argparse
import random
import pandas as pd
import matplotlib.pyplot as plt
import umap

from ptb import *
from model import SentenceVAE
from utils import to_var, idx2word, interpolate, similarity
from data_select_lib import *

def main(args):

    torch.manual_seed(0)

    # load model
    with open(f'{args.data_dir}/{args.data_prefix}.vocab.json', 'r') as file:
        vocab = json.load(file)


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
        cond_size=0
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

    ##############################################################################################


    # load tunes from split

    data = PTB(
        data_dir=args.data_dir,
        create_data=False,
        split=args.split,
        max_sequence_length=args.max_sequence_length,
        data_prefix=args.data_prefix,
        conditioned=args.conditioned,
        bars=args.bars
    )

    # select stratified data
    dic = stratified_bar_select(args.num_samples, args.type, data)

    for k in dic:
        print(k, len(dic[k]))

    ##############################################################################################

    # create bar embeddings

    points = {}

    for key in dic.keys():
        points[key] = []

        for index in dic[key]:

            tune = torch.tensor(np.array([data[index]['input']]))
            length = [data[index]['length']]

            if cuda:
                logp, mean, logv, z, z_cond = model(tune.cuda(),
                                                    torch.tensor(length).cuda(),
                                                    None)
            else:
                logp, mean, logv, z, z_cond = model(tune,
                                                    torch.tensor(length),
                                                    None)
            z = z.detach().cpu().numpy()[0]
            points[key].append(z)

    ##############################################################################################

    # create tune embeddings

    with open(f'data/bars_no_key_conditioned.vocab.json', 'r') as file:
        bar_vocab = json.load(file)
    with open(f'data/data_v2_cleaned.{args.split}.json') as f:
        data = json.load(f)
    with open(f'data/data_v2_cleaned.vocab.json', 'r') as file:
        tune_vocab = json.load(file)

    bar_w2i, bar_i2w = bar_vocab['w2i'], bar_vocab['i2w']
    tune_w2i, tune_i2w = tune_vocab['w2i'], tune_vocab['i2w']

    # pick random tune
    tune_num = 69#random.randint(0, len(data))
    tune = torch.tensor(np.array([data[str(tune_num)]['input']]))
    tune_str = idx2word(tune, i2w=tune_i2w, pad_idx=tune_w2i['<pad>'])[0]

    print()
    print('Processing the following tune.')
    print(tune_str)
    print()

    # save key & time for later
    time = tune_str.split(' ')[1]
    key = tune_str.split(' ')[2]

    # append to output
    tune_str = ' '.join(tune_str.split(' ')[3:])

    # process bars
    if tune_str[0] != '|':
        tune_str = '| ' + tune_str

    raw_bars = tune_str.split('|')
    # remove ''
    while '' in raw_bars:
        raw_bars.remove('')
    # add | again to each bar
    raw_bars = [f'|{b}|' for b in raw_bars]

    print(raw_bars)

    # translate tokens to indexes
    bars = []
    for b in raw_bars:
        tmp = []
        for wd in b.split(' '):
            tmp.append(bar_w2i[wd])
        bars.append(tmp)


    points['tune'] = []

    # for each bar
    for bar in bars:

        # model input
        tensor = torch.tensor(np.array([bar]))

        if args.seed:
            torch.manual_seed(0)

        # forward
        logp, mean, logv, z, z_cond = model(tensor.cuda(),
                                            torch.tensor([len(bar)]).cuda(),
                                            None)

        z = z.detach().cpu().numpy()[0]
        points['tune'].append(z)


    ##############################################################################################

    # prepare for umap

    # all embeddings
    all_points = {}
    whole = {str(i): [] for i in range(args.latent_size)}

    # create pandas data frame for umap
    for k in points.keys():
        # init dimensions cols
        cat_df = {str(i): [] for i in range(args.latent_size)}

        for z in points[k]:

            # append value to each dim
            for (i, j) in enumerate(z):
                # append to value-specific dic
                cat_df[str(i)].append(j)
                # append to complete dic
                whole[str(i)].append(j)

        # attach to all points
        cat_df = pd.DataFrame(cat_df)
        all_points[k] = cat_df

    whole = pd.DataFrame(whole)


    ##############################################################################################

    # run umap

    reducer = umap.UMAP(random_state=42)
    reducer.fit(whole)

    for i, value in enumerate(all_points.keys()):
        if args.latent_size != 2:
            embedding = reducer.transform(all_points[value])

        else:
            embedding = np.array([
                [
                    all_points[value]['0'].iloc[j],
                    all_points[value]['1'].iloc[j]
                ]
                for j in range(len(all_points[value]['0']))
            ])

        try:
            lbl = i2w[str(value)]
        except:
            lbl = value



        if value == 'tune':
            plt.plot(embedding[:, 0],
                    embedding[:, 1],
                    color='black',
                    label=lbl,
                    marker='.')

            if not args.seed:
                for l in [('*', 0), ('X', -1)]:

                    marker= l[0]
                    index = l[1]

                    x = embedding[index,0]
                    y = embedding[index,1]

                    plt.plot(x, y, markersize=15, marker=marker,
                             markeredgecolor='#000000',
                             markerfacecolor='#ffff00')

        else:
            plt.scatter(embedding[:, 0],
                        embedding[:, 1],
                        cmap='Spectral',
                        label=lbl, s=15,
                        marker='o')


    plt.gca().set_aspect('equal', 'datalim')
    plt.legend()

    title = f'folkbar-VAE {args.latent_size}/{args.hidden_size} projection of tune {args.split}-{tune_num}'

    if args.latent_size != 2:
        title += ' (UMAP projection)'


    plt.suptitle(title, fontweight='bold')


    name = args.type
    if args.type == 'meter':
        name = 'time'

    s = f'{args.num_samples} bars per {name}'
    if args.type != 'function':
        s += ' signature'
    if args.seed:
        s += ' - seeded'

    plt.title(s)

    plt.savefig(
        f'bars_no_cc/plots/folkbar-VAE_{args.latent_size}-{args.hidden_size}_{args.type}__{args.seed}_tune.png',
        dpi=200
    )

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
    parser.add_argument('-tp', '--type', type=str, default='key')
    parser.add_argument('-sd', '--seed', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.mode = args.mode.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.mode in ['greedy', 'topk', 'topp']
    assert args.split in ['train', 'test', 'valid']
    assert 0 <= args.word_dropout <= 1

    main(args)
