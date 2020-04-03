import os
import pandas as pd
import multiprocessing as mp
import gc

excludeFiles = ['Datasets.md']
chunklines = 1000000


def processFile(filename):
    print('Starting: ' + filename)
    chunks = pd.read_json('./raw_data/' + filename,
                          lines=True, chunksize=chunklines, compression='gzip')
    for df in chunks:
        # Only Keep Columns We Care About
        df = df[['summary', 'reviewText']]
        gc.collect()
        # Dimensionality Reduction
        df['review'] = df['summary'].astype(
            str) + ' ' + df['reviewText'].astype(str)
        # Labeling
        df['category'] = filename[0:-8]
        df = df[['review', 'category']]
        gc.collect()
        # Convert to Lowercase (agressively)
        df['review'] = df['review'].str.casefold()
        # Convert all whitespace to space
        df['review'] = df['review'].str.replace('[\s]', ' ')
        # Remove all punctuation except for single quote
        df['review'] = df['review'].str.replace("[^\w\s']", "")

        df.to_csv('./datasets/' +
                  filename[0:-8] + '.csv', mode='a+', index=False)
    print('Finished: ' + filename)


if __name__ == '__main__':
    # Note: This should be (at most) the number of physical cores
    pool = mp.Pool(processes=8)
    # ^Reducing this also reduces memory consumption
    for filename in os.listdir('./raw_data'):
        if (filename not in excludeFiles):
            pool.apply_async(processFile, args=(filename,))
    pool.close()
    pool.join()
