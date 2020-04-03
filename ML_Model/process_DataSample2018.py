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


def processFile(filename, queue):
    print('Importing / Labeling: ' + filename)
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
    df_file = df_file.sample(n=nKeep)
    gc.collect()
    queue.put(df_file)


if __name__ == '__main__':
    df_sample = []
    q = mp.Queue()
    for filename in os.listdir('./datasets'):
        if (filename not in excludeFiles):
            p = mp.Process(target=processFile, args=(filename, q))
            p.start()
            df_sample.append(q.get())
    print('Creating Dataframe')
    df_sample = pd.concat(df_sample)
    # Unlikely, But Check for Duplicates Again - Drop All Cases Due to Ambiguity
    df_sample.drop_duplicates(
        subset=['review'], keep=False, inplace=True)
    print("Writing to Output")
    gc.collect()
    df_sample.to_csv('out_sample_all' + str(args.n) +
                     '.csv', header=False, index=False)
