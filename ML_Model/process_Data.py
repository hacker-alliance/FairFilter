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

for filename in os.listdir('./datasets'):
    if (filename != 'Datasets.md'):
        fdf = getDF('./datasets/' + filename)
        dfs.append(fdf)

df = pd.concat(dfs)

df.to_csv('out.csv', header=False, index=False)
