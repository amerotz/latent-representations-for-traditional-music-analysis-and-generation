import os
import numpy as np
import json
import matplotlib.pyplot as plt

fdir = 'bars_no_cc/plots/training'

for file in sorted(os.listdir(fdir+'/valid'), key=lambda x : int(x.split('_')[7].split('=')[-1])):

    if not '.json' in file:
        continue

    with open(f'{fdir}/valid/{file}', 'r') as f:

        name = file.split('_')
        name = name[7].split('=')[-1] + '/' + name[4].split('=')[-1]

        data = json.load(f)
        data = [d[2] for d in data]
        plt.plot(data, label=name, marker='.')

plt.title('folkbar-VAE validation ELBO over epochs', fontweight='bold')
plt.legend()
plt.xticks(range(0,21))
plt.ylim(14)

plt.savefig(f'{fdir}/folkbar-VAE_valid.png', dpi=200)
'''

fdir = 'full_tunes/plots/training'

for file in sorted(os.listdir(fdir+'/kl'), key=lambda x : int(x.split('_')[7].split('=')[-1])):

    if not '.json' in file:
        continue

    with open(f'{fdir}/kl/{file}', 'r') as f:

        name = file.split('_')
        name = name[7].split('=')[-1] + '/' + name[4].split('=')[-1]

        data = json.load(f)
        data = [d[2] for d in data]
        epoch = len(data)//20
        new_data = [data[min(epoch*i, epoch*20-1)] for i in range(21)]
        #new_data = [np.sum(data[:epoch*(i+1)])/(epoch*(i+1)) for i in range(21)]
        plt.plot(new_data, label=name, marker='.')

plt.title('folktune-VAE validation true KL-Divergence over epochs', fontweight='bold')
plt.legend()
plt.xticks(range(0,21))
plt.yscale('log')

plt.savefig(f'{fdir}/folktune-VAE_true_kl.png', dpi=200)

'''

