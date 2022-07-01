import os


for i in range(1, 6):

    ls = 2**i
    hs = ls*8
    print('Training new model:', ls, hs)
    os.system(f'python3 train.py -tb --data_dir data -dp data_v2_cleaned -ep 21 -wd 0 -ed 0 -rnn gru --max_sequence_length 256 -bs 32 --num_layers 2 --hidden_size {hs} --latent_size {ls} --save_model_path full_tunes --logdir full_tunes/logs --test')
