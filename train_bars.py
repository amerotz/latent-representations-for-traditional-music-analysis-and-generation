import os


for i in range(1, 6):

    ls = 2**i
    hs = ls*8
    print('Training new model:', ls, hs)
    os.system(f'python3 train.py -tb --data_dir data -dp bars_no_key_conditioned -ep 21 -wd 0 -ed 0 -rnn gru --max_sequence_length 32 -bs 32 --num_layers 2 --hidden_size {hs} --latent_size {ls} -bb --save_model_path bars_cc --logdir bars_cc/logs --test -cc')
