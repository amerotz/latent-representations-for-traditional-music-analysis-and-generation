import argparse
import numpy as np
import re

parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()

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

with open(args.file, 'r') as f:
    data = f.read()

data = data.split('\n')

disc = 0

new_data = []
for d in data:

    tmp = d.split(' ')

    key = tmp[1]
    time = tmp[0]

    tmp = tmp[2:]
    tmp = ' '.join(tmp)
    tmp = tmp.split('|')

    while ' ' in tmp:
        tmp.remove(' ')

    while '' in tmp:
        tmp.remove('')

    l = [f'| {x.strip()} |\t{key}\t{time}\t{function(x)}' for x in tmp if len(x.split(' ')) < 32 and x != ':' ]

    new_data.extend(l)


new_data = '\n'.join(new_data)
new_data = new_data.replace(': |', ':|')
new_data = new_data.replace('| :', '|:')
new_data = new_data.replace('| 1', '|1')
new_data = new_data.replace('| 2', '|2')


with open('bars_no_key_conditioned.txt', 'w') as f:
    f.write(new_data.strip())


