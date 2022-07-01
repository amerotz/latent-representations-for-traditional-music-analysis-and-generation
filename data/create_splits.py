import argparse
import random
import json

parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()


TRAIN = 0.8
VALID = 0.1
TEST = 0.1


with open(args.file + '.txt', 'r') as f:
    data = f.read()

data = data.split('\n')


train, valid, test = [], [], []

for tune in data:

    n = random.random()

    if n <= TRAIN:
        train.append(tune)
    elif n < TRAIN + VALID:
        valid.append(tune)
    else:
        test.append(tune)

with open(f'{args.file}.train.txt', 'w') as f:
    f.write('\n'.join(train))


with open(f'{args.file}.valid.txt', 'w') as f:
    f.write('\n'.join(valid))

with open(f'{args.file}.test.txt', 'w') as f:
    f.write('\n'.join(test))
