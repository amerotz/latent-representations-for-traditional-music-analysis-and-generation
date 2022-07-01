import os
import json

fdir = 'bars_no_cc/models'

# for each model
for model in os.listdir(fdir):

    with open(f'{fdir}/{model}/model_params.json', 'r') as f:
        params = json.load(f)

    ls = params['latent_size']
    hs = params['hidden_size']

    for t in ['key', 'meter', 'function']:

        '''
        print('Bars', t, ls, hs)
        # plot bars 
        os.system(
            f'python3 folkbar_plots.py -c {fdir}/{model}/best.pytorch -hs {hs} -nl 2 -ls {ls} -sp test -dp bars_no_key_conditioned --data_dir data -ms 32 -bb -n 200 -tp {t} --seed'
        )
        print()


        print('Tunes', t, ls, hs)
        # plot tunes
        os.system(
            f'python3 folkbar_tune.py -c {fdir}/{model}/best.pytorch -hs {hs} -nl 2 -ls {ls} -sp test -dp bars_no_key_conditioned --data_dir data -ms 32 -bb -n 200 -tp {t} --seed'
        )
        print()
        '''

        os.system(
            f'python3 folkbar_tune_hist.py -c {fdir}/{model}/best.pytorch -hs {hs} -nl 2 -ls {ls} -sp test -dp bars_no_key_conditioned --data_dir data -ms 32 -bb'
        )
        os.system(
            f'python3 folkbar_tune_hist.py -c {fdir}/{model}/best.pytorch -hs {hs} -nl 2 -ls {ls} -sp test -dp bars_no_key_conditioned --data_dir data -ms 32 -bb --seed'
        )
