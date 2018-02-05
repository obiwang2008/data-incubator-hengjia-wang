import sys
import numpy as np
import pandas as pd
from sklearn import *
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def predict(comment):
    pkl_file = open('model.pkl', 'rb')
    model = pickle.load(pkl_file)
    train = pd.read_csv('input/jigsaw-toxic-comment-classification-challenge/train.csv', nrows = 39892)
    test = pd.read_csv('input/jigsaw-toxic-comment-classification-challenge/test.csv', nrows = 38291)
    df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
    df = df.fillna("unknown")
    tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english', max_features=800000)
    data = tfidf.fit_transform(df)
    test_data = tfidf.transform([comment])
    sub2 = model.predict_proba(test_data)
    sub2 = pd.DataFrame([[c[1] for c in sub2[row]] for row in range(len(sub2))]).T
    return sub2.values.tolist()[0]

#########################################
if __name__ == "__main__":
    comment = sys.argv[1]
    results=predict(comment)
