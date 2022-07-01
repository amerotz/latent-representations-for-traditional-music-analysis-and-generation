from ptb import *

def stratified_select(num, cat, data):

    index = 'a'

    if cat == 'key':
        index = 2
    elif cat == 'time':
        index = 1

    dic = {}

    for i in range(len(data)):

        piece = data[i]['input']
        tok = piece[index]

        if tok in dic:
            if len(dic[tok]) < num:
                dic[tok].append(i)
        else:
            dic[tok] = [i]

    return dic

def stratified_bar_select(num, cat, data):

    dic = {}

    for i in range(len(data)):

        tok = data[i][cat]

        if tok in dic:
            if len(dic[tok]) < num:
                dic[tok].append(i)
        else:
            dic[tok] = [i]

    return dic

def index_stratified_bar_select(num, index, data):

    dic = {}

    for i in range(len(data)):

        tok = str(data[i]['input'][:index])

        if tok in dic:
            if len(dic[tok]) < num:
                dic[tok].append(i)
        else:
            dic[tok] = [i]

    return dic



'''
# load tunes from split

data = PTB(
    data_dir='data',
    create_data=False,
    split='test',
    max_sequence_length=256,
    data_prefix='data_v2_cleaned',
    conditioned=False,
    bars=False
)

d = stratified_select(100, 'key', data)
print(d)

with open(f'data/data_v2_cleaned.vocab.json', 'r') as file:
    vocab = json.load(file)

w2i, i2w = vocab['w2i'], vocab['i2w']

for x in d.keys():
    print(i2w[str(x)])
'''
