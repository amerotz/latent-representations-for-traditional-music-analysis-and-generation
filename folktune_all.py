import os
import json

fdir = 'full_tunes/models'

# for each model
for model in os.listdir(fdir):

    with open(f'{fdir}/{model}/model_params.json', 'r') as f:
        params = json.load(f)

    ls = params['latent_size']
    hs = params['hidden_size']

    for t in ['key', 'time']:

        print('Tunes', t, ls, hs)
        # plot tunes
        os.system(
            f'python3 folktune_plots.py -c {fdir}/{model}/E20.pytorch -hs {hs} -nl 2 -ls {ls} -sp train -dp data_v2_cleaned --data_dir data -pr -ms 256 -n 200 -tp {t}'
        )
        print()

        print('Reels', t, ls, hs)
        # plot reels
        os.system(
            f'python3 folktune_reels.py -c {fdir}/{model}/E20.pytorch -hs {hs} -nl 2 -ls {ls} -sp train -dp data_v2_cleaned --data_dir data -pr -ms 256 -n 200 -tp {t}'
        )
        print()
