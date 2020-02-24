import pandas as pd
import gzip
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

def getCombinedDF():
    dfs = []
    for filename in os.listdir('./datasets'):

        if (filename != 'Datasets.md'):
            fdf = getDF('./datasets/' + filename)
            needed = pd.DataFrame({'review': (fdf['summary'] + ". " + fdf['reviewText']).replace(',', ' '), 'category': filename[8:-10]})
            dfs.append(needed)

    df = pd.concat(dfs)
    return df

def getSampleDF(numReviewsPerCategory):
    dfs_sample = []

    for filename in os.listdir('./datasets'):
        if (filename != 'Datasets.md'):
            fdf = getDF('./datasets/' + filename)
            needed = pd.DataFrame({'review': (fdf['summary'] + ". " + fdf['reviewText']).replace(',', ' '), 'category': filename[8:-10]})
            sample_rows = needed.sample(n = numReviewsPerCategory)
            dfs_sample.append(sample_rows)

    df_sample = pd.concat(dfs_sample)
    df_sample.reset_index(inplace=True)
    return df_sample



def pipeline(vectorizer_features, df):
    pipe2 = Pipeline([("vectorizer", TfidfVectorizer(max_features=vectorizer_features,stop_words='english')), ("model", LogisticRegression(C=1))])
    X_train, X_test, y_train, y_test = train_test_split(df.review,df.category,train_size=.1)
    pipe2.fit(X_train,y_train)
    return pipe2, X_train, X_test, y_train, y_test


df = getCombinedDF()
df_sample = getSampleDF(10000)
pipe, X_train, X_test, y_train, y_test = pipeline(10000, df_sample)


# to get accuracy:
# accuracy_score(y_test,rf.predict(vec[y_test.index,:]))










