import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.sentiment import sentiment_analyzer
import nltk
import json
import graphlab
from graphlab import SFrame
from scipy.sparse import dok_matrix
from sklearn.decomposition import TruncatedSVD
import math

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

def extract_topics_nmf(text, n_topics = 5):
    tfidf, tfidf_vectorizer = fit_tfidf(text)

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

def matrix_to_sframe(tfidf):
    '''
    When passed a numpy sparse matrix, the function converts it into an SFrame.
    '''
    reviews, features, vals = [], [], []

    for ft_idx in range(tfidf.shape[1]):
        feature = tfidf.getcol(ft_idx).todok()
        keys = feature.keys()
        reviews += [key[0] for key in keys]
        features += [key[1] for key in keys]
        vals += feature.values()

    return SFrame({'feature':features, 'review':reviews, 'tfidf':vals})



def build_recommender(sf, reduced_categories):
    #Convert reduced_categories to an SFrame, then add the review numbers
    reduced_review_category = SFrame(pd.DataFrame(reduced_categories))
    reduced_review_category.add_column(graphlab.SArray(df.index.get_values()), name='review')

    rec = graphlab.recommender.factorization_recommender.create( sf,
    user_id='review', item_id='feature', target='tfidf', user_data = reduced_review_category, solver='adagrad', side_data_factorization = True)
    return rec

def predict_one(rec, datapoint = SFrame({'feature': [4350], 'review': [430800]}) ):
    prediction = "rating:", rec.predict(one_datapoint)[0]
    print prediction
    return prediction

def extract_side_features():
    side_features = has_reviews[['reviewTime', 'categories', 'price', 'salesRank']]
    for col in side_features.columns:
        side_features[col] = str()
    review_data = SFrame(side_features)

    side_features = has_reviews[['reviewTime', 'categories', 'price', 'salesRank']]

def extract_key_categories(df):
    cats = df['categories']
    clean_cats = cats.apply(clean_categories)
    categories = unique_categories(clean_cats)
    item_category_mat = dok_matrix((clean_cats.shape[0], len(categories)))

    for cat_idx, category in enumerate(clean_cats):
        for item in category:
            idx = categories.index(item)
            item_category_mat[cat_idx, idx] = 1

    # # 55 Components was chosen as the optimal value, explaining 55.03% of the variance, with additional components offer little additional explanatory value.

    # explained_variances = {}
    # for n in np.arange(5, 200, 10):
    #     model = TruncatedSVD(n_components=n, n_iter = 10)
    #     model.fit(item_category_mat)
    #     print(model.explained_variance_ratio_)
    #     explained_variances[n] = model.explained_variance_ratio_
    #
    # for n in sorted(explained_variances.keys()):
    #     print str(n) + " components explained " + str(round(np.sum(explained_variances[n])*100,2))+ "% of the variance"

    model = TruncatedSVD(n_components=55, n_iter = 30).fit(item_category_mat)
    return model, model.transform(item_category_mat)


def clean_salesRank_col():
    # #25% missing values
    # test = df.salesRank[0]
    # df.salesRank.apply(clean_salesRank)
    #
    # #Got it
    # df.salesRank.fillna("Blank: 0").replace('{}', "Blank: 0").value_counts()
    #
    # str(df.salesRank).strip("{}'").split(': ')[1]
    #
    # df.salesRank.fillna("Blank: 0").apply(clean_salesRank)
    #
    # ("Blank: 0").strip("{}'").split(': ')[1])

def clean_salesRank_row(row, fill_val = 0):
    # if math.isnan(row) or type(row) is not str:
    #     return fill_val
    try:
        broad_cat = str(row).strip("{}'").split(': ')[0].strip("'")
        rank = int(str(row).strip("{}'").split(': ')[1])
        return (broad_cat, rank)
    except TypeError, IndexError:
        return (fill_val, fill_val)

    # broad_cat = df.salesRank.apply(lambda x: str(x).strip("{}'").split(': ')[0].strip("'"))
    # rank  = df.salesRank.apply(lambda x: (str(x).strip("{}'").split(': '))[1])

def unique_categories(series):
    '''
    Returns a sorted list with the unique categories in the series.
    '''
    categories = []
    for row in series:
        for cat in row:
            categories.append(cat)
    return sorted(list(set(categories)))

def clean_categories(row):
    '''
    When passed a single row of categories, removes all unnecessary whitespace, nested loops, and additional quotation marks and returns a single list of categories.
    '''
    items = str(row).strip('[]').split(",")
    cleaned_items = []
    for item in items:
        temp = item.strip("' [],'")
        cleaned_items.append(temp)
    return cleaned_items

if __name__ == '__main__':
    df = pd.read_csv('cleaned_one_star.csv')
    reviews = df.reviewText.values
    #tfidf, tfidf_vectorizer = extract_topics_nmf(reviews)
    tsvd_model, reduced_categories = extract_key_categories(df)


    #sf = matrix_to_sframe(tfidf)
    sf = graphlab.load_sframe('tfidf_sframe.csv')
    rec = build_recommender(sf, reduced_categories)
