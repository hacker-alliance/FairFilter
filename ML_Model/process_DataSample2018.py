import pandas as pd
import gzip
import os
import json
import argparse

chunkRows = 1000000


def getDFChunks(path):  # Get Dataframe After Parsing
    return pd.read_json(path, lines=True, chunksize=chunkRows)


parser = argparse.ArgumentParser(description='Process Amazon Review Dataset')
parser.add_argument('n', metavar='n', type=int, help='Sample Size')
args = parser.parse_args()

df_sample = []
excludeFiles = ['Datasets.md', 'Books.json', 'Gift_Cards.json', 'Magazine_Subscriptions.json', 'CDs_and_Vinyl.json',
                'Movies_and_TV.json', 'Video_Games.json', 'Kindle_Store.json', 'Digital_Music.json', 'Software.json']

for filename in os.listdir('./datasets'):
    if (filename not in excludeFiles):
        df_file = []
        print('Importing / Labeling: ' + filename)
        for df in getDFChunks('./datasets/' + filename):
            # Memory Optimization - Keep only relevant columns
            df = df[['summary', 'reviewText']]
            df['category'] = filename[0:-5]
            df_file.append(df)

        df_file = pd.concat(df_file)
        # Dimensionality Reduction - Combine title (summary) and reviewText
        df_file['review'] = df_file['summary'] + ' ' + df_file['reviewText']
        df_file = df_file[['category', 'review']]
        # Cleaning / More Memory Optimization - Drop Duplicates
        df_file.drop_duplicates(
            subset=['review'], keep='first', inplace=True)
        print(df_file.tail())
        print('Number of Reviews: ' + str(len(df_file.index)))
        df_file = df_file.sample(n=args.n)
        df_sample.append(df_file)

print('Creating Dataframe')
df_sample = pd.concat(df_sample)
# Unlikely, But Check for Duplicates Again - Drop All Cases Due to Ambiguity
df_sample.drop_duplicates(
    subset=['review'], keep=False, inplace=True)
print("Writing to Output")
df_sample.to_csv('out_sample' + str(args.n) +
                 '.csv', header=False, index=False)
