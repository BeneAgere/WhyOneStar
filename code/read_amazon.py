import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.sentiment import sentiment_analyzer
import nltk
import json
import graphlab

def top_features(vec, n = 10):
    feature_array = np.array(vec.get_feature_names())
    indices = np.argsort(vec.idf_)[::-1]
    features = vec.get_feature_names()
    top_features = [features[i] for i in indices[:n]]
    return top_features

def query(df, query):
    data = fetch_20newsgroups(subset='train', categories=categories).data
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(data).toarray()
    words = vectorizer.get_feature_names()

    tokenized_queries = vectorizer.transform(queries)
    cosine_similarities = linear_kernel(tokenized_queries, vectors)
    titles = newsgroups.filenames
    for i, query in enumerate(queries):
        print(query)
        print(get_top_values(cosine_similarities[i], 3, titles))


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
            for i in topic.argsort()[ :-n_top_words - 1:-1]]))
    print()

def fit_tfidf(text):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, max_features = 5000)
    tfidf = vectorizer.fit_transform(text)
    return tfidf, vectorizer

def topics_extraction(text, n_topics = 5):
    nmfModel = NMF(n_components=5, random_state=1, alpha=.1, l1_ratio=.5)
    nmfModel.fit(tfidf)

    tfidf = tfidf_vectorizer.fit_transform(one_star.reviewText.values)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    tfidf_feature_names
    print_top_words(nmf, tfidf_feature_names, 20)

    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf = vec.fit_transform(text)
    nmfModel = NMF(n_topics)
    W = nmfModel.fit_transform(tfidf)
    pass

def remove_nan_reviews(data):
    nan_indices = []
    lengths = []
    for index, line in enumerate(data):
        try:
            lengths.append(len(line))
        except TypeError:
            nan_indices.append(index)
    print("Error reading {} lines".format(len(nan_indices)))
    return np.delete(data, nan_indices)

def extract_topics_nmf(text, n_topics = 5):
    tfidf, tfidf_vectorizer = fit_tfidf(text)

    # Fit the NMF model
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

    print("\nTopics in NMF model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words=10)
    return tfidf, tfidf_vectorizer

def dirichlet(text, n_topics = 5):
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words='english')
    tf = tf_vectorizer.fit_transform(text)

    lda = LatentDirichletAllocation( n_topics=n_topics, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
    lda.fit(tf)

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

def lemmatize(sentence):
    from textblob import TextBlob
    tb = Textblob(sentence)
    return list(tb.words.lower().lemmatize())

def analyze_sentiment(df):
    from textblob import TextBlob
    reviews = df.reviewText.values
    polarity_scores = []
    subjectivity_scores = []
    for review in reviews:
        tb = TextBlob(review)
        polarity_scores.append(tb.sentiment.polarity)
        subjectivity_scores.append(tb.sentiment.subjectivity)
    return polarity_scores, subjectivity_scores

def vader(df):
    sentiments = []
    #one_star.reviewText.apply(nltk.sentiment.util.demo_liu_hu_lexicon)
    #nltk.sentiment.util.demo_liu_hu_lexicon(df.reviewText.iloc[0])
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk import tokenize
    test = tokenize.sent_tokenize(reviews[0])


    tot_neg, tot_pos = [], []
    for sentence in review_list:
        print(sentence)
        ss = sid.polarity_scores(sentence)
        for k in sorted(ss):
            print('{0}: {1}, '.format(k, ss[k]))
            print()

    print(test)
    ss = sid.polarity_scores(test)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]))
        print()

def graphlab_fun(df):
    reviews, features, vals = [], [], []

    for ft_idx in range(tfidf.shape[1]):
        feature = tfidf.getcol(ft_idx).todok()
        keys = feature.keys()
        reviews += [key[0] for key in keys]
        features += [key[1] for key in keys]
        vals += feature.values()

    sf = SFrame({'feature':features, 'review':reviews, 'tfidf':vals})

    rec = graphlab.recommender.factorization_recommender.create(sf,
    user_id='review', item_id='feature', target='tfidf', solver='als',
    side_data_factorization=False)


def add_metadata(df):
    meta = pd.read_csv('metadata.csv')
    return pd.merge(df, meta, on='asin')

if __name__ == '__main__':
    df = pd.read_csv('one_star.csv')
    reviews = remove_nan_reviews(df.reviewText.values)
    tfidf, tfidf_vectorizer = extract_topics_nmf(reviews)

    #add_metadata(df)
    #print top_features(vec, 10)
