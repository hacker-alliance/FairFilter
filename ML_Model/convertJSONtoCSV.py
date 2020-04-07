import os
import pandas as pd
import re
import multiprocessing as mp
import gc

excludeFiles = ['Datasets.md']
chunklines = 1000000

whitespaceExp = re.compile('\s\s+')
punctuationExp = re.compile("[^\w\s']")
leadTrailExp = re.compile("^\s+|\s+$")


def processFile(filename):
    print('Starting: ' + filename)
    chunks = pd.read_json('./raw_data/' + filename,
                          lines=True, chunksize=chunklines)
    for df in chunks:
        # Only Keep Columns We Care About
        df = df[['summary', 'reviewText']]
        # Dimensionality Reduction
        df['review'] = df['summary'].astype(
            str) + ' ' + df['reviewText'].astype(str)
        df = df[['review']]
        # Convert to Lowercase
        df['review'] = df['review'].str.lower()
        # Remove all punctuation aexcept for single quote
        df['review'] = df['review'].str.replace(
            punctuationExp, "")
        # Convert all contiguous whitespace to a single space
        df['review'] = df['review'].str.replace(whitespaceExp, ' ')
        # Remove Leading / Trailing whitespace
        df['review'] = df['review'].str.replace(
            leadTrailExp, "")
        # Labeling
        df['category'] = filename[0:-5]
        df.to_csv('./datasets/' +
                  filename[0:-5] + '.csv', mode='a+', index=False)
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
