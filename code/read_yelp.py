import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.sentiment import sentiment_analyzer

def read_json(filepath):
    with open(filepath) as f:
        data = f.readlines()
    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ",".join(data) + "]"
    return pd.read_json(data_json_str)

def tfidf(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(data).toarray()
    words = vectorizer.get_feature_names()
    return vectorizer

def top_features(vec, n = 10):
    feature_array = np.array(vec.get_feature_names())
    indices = np.argsort(vec.idf_)[::-1]
    features = vec.get_feature_names()
    top_features = [features[i] for i in indices[:n]]
    return top_features

def query(query):
    data = fetch_20newsgroups(subset='train', categories=categories).data
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(data).toarray()
    words = vectorizer.get_feature_names()

    # get queries from file
    with open('data/queries.txt') as f:
        queries = [line.strip() for line in f]

    tokenized_queries = vectorizer.transform(queries)
    cosine_similarities = linear_kernel(tokenized_queries, vectors)
    titles = newsgroups.filenames
    for i, query in enumerate(queries):
        print query
        print get_top_values(cosine_similarities[i], 3, titles)
        print

def topics_extraction(tfidf, n_topics = 5):
    print("Fitting the NMF model with tf-idf features")
    return NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

if __name__ == '__main__':
    df = read_json('yelp_academic_dataset_review.json')
    one_star = df[df.overall == 1]
    vec = tfidf(one_star.reviewText.values)

    print top_features(vec, 10)
