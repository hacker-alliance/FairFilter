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
dfs_sample = []

for filename in os.listdir('./datasets'):
    if (filename != 'Datasets.md'):
        fdf = getDF('./datasets/' + filename)
        needed = pd.DataFrame({'review': (fdf['summary'] + ". " + fdf['reviewText']).replace(',', ' '), 'category': filename[8:-10]})
        # sample100 = needed.sample(n = 100)
        dfs.append(needed)
        # dfs_sample.append(sample100)

df = pd.concat(dfs)
df.to_csv('out.csv', header=False, index=False)

# comment out ines 21-31 if you ONLY need to grab the sample.
# if you need to grab both, comment the for loop out and uncomment the commented
# lines in the above for loop
for filename in os.listdir('./datasets'):
    if (filename != 'Datasets.md'):
        fdf = getDF('./datasets/' + filename)
        needed = pd.DataFrame({'review': (fdf['summary'] + ". " + fdf['reviewText']).replace(',', ' '), 'category': filename[8:-10]})
        sample100 = needed.sample(n = 100)
        dfs_sample.append(sample100)

df_sample = pd.concat(dfs_sample)
df_sample.to_csv('out_sample.csv', header=False, index=False)
