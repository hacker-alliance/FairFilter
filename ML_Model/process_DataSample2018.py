import pandas as pd
import gzip
import os
import json
import argparse
import gc
import multiprocessing as mp

parser = argparse.ArgumentParser(description='Process Amazon Review Dataset')
parser.add_argument('n', metavar='n', type=int, help='Sample Size')
args = parser.parse_args()

excludeFiles = ['Datasets.md']


def processDataset(filename, queue):
    print('Importing: ' + filename)
    df_file = pd.read_csv('./datasets/' + filename, dtype=str,
                          usecols=['review', 'category'])
    # Cleaning - Drop Duplicates
    df_file.drop_duplicates(
        subset=['review'], keep='first', inplace=True)
    print(df_file.tail())
    l = len(df_file.index)
    print('Number of Reviews: ' + str(l))
    nKeep = args.n
    if args.n > l:
        nKeep = l
        print('Warning: Requested More Samples Than Available')
    else:
        df_file = df_file.sample(n=nKeep)

    df_file.to_csv('./out/sample_' + str(round(args.n/1000)) +
                   'k.csv', header=False, index=False, mode='a+')


def finalPass():
    print('Starting Final Pass')
    df_file = pd.read_csv('./out/sample_' + str(round(args.n/1000)) +
                          'k.csv', dtype=str,
                          names=['review', 'category'])
    # Drop Duplicates - Don't Keep Duplicates Due to Ambiguity
    df_file.drop_duplicates(
        subset=['review'], keep=False, inplace=True)
    df_file.to_csv('./out/sample_' + str(round(args.n/1000)) +
                   'k.csv', header=False, index=False, mode='w')
    print('Finished Final Pass')


if __name__ == '__main__':
    df_sample = []
    q = mp.Queue()
    for filename in os.listdir('./datasets'):
        if (filename not in excludeFiles):
            p = mp.Process(target=processDataset, args=(filename, q))
            p.start()
            p.join()
    finalPass()
