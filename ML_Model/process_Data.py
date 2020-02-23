import pandas as pd
import gzip
import os


def parse(path):  # Unzip and Parse JSON
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def getDF(path):  # Get Dataframe After Parsing
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


dfs = []
dfs_sample100 = []
dfs_sample1k = []
dfs_sample10k = []

for filename in os.listdir('./datasets'):
    if (filename != 'Datasets.md'):
        fdf = getDF('./datasets/' + filename)
        needed = pd.DataFrame({'review': (
            fdf['summary'] + ". " + fdf['reviewText']).replace(',', ' '), 'category': filename[8:-10]})
        needed.drop_duplicates(keep=False, inplace=True)
        dfs.append(needed)

        sample100 = needed.sample(n=100)
        dfs_sample100.append(sample100)

        sample1k = needed.sample(n=1000)
        dfs_sample1k.append(sample1k)

        sample10k = needed.sample(n=10000)
        dfs_sample10k.append(sample10k)

df = pd.concat(dfs)
df.to_csv('out_full.csv', header=False, index=False)

df_sample100 = pd.concat(dfs_sample1k)
df_sample100.to_csv('out_sample100.csv', header=False, index=False)

df_sample1k = pd.concat(dfs_sample1k)
df_sample1k.to_csv('out_sample1k.csv', header=False, index=False)

df_sample10k = pd.concat(dfs_sample10k)
df_sample10k.to_csv('out_sample10k.csv', header=False, index=False)
