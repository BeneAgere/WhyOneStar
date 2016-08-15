import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.sentiment import sentiment_analyzer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
import nltk
import json
import graphlab
from graphlab import SFrame
from scipy.sparse import dok_matrix
from sklearn.decomposition import TruncatedSVD
import math
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from time import time

def top_features(vec, n = 10):
    feature_array = np.array(vec.get_feature_names())
    indices = np.argsort(vec.idf_)[::-1]
    features = vec.get_feature_names()
    top_features = [features[i] for i in indices[:n]]
    return top_features

def query(df, query):
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

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def fit_vectorizer(text, kwargs, type='tfidf'):
    if type == 'tfidf':
        vectorizer = TfidfVectorizer(**kwargs)
        counts = vectorizer.fit_transform(text)
    elif type == 'count':
        vectorizer = CountVectorizer(**kwargs)
        counts = vectorizer.fit_transform(text)
    else:
        counts, vectorizer = [], []
    return counts, vectorizer

def topics_extraction(text, n_topics = 5):
    nmfModel = NMF(n_components=5, random_state=1, alpha=.1, l1_ratio=.5)
    nmfModel.fit(tfidf)

    tfidf = tfidf_vectorizer.fit_transform(one_star.reviewText.values)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    tfidf_feature_names
    print_top_words(nmf, tfidf_feature_names, 20)

    tfidf = vec.fit_transform(text)
    nmfModel = NMF(n_topics)
    W = nmfModel.fit_transform(tfidf)
    pass

def print_topics_nmf(tfidf, n_topics = 5):
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    print("\nTopics in NMF model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words=10)
    pass

def dirichlet(text, n_topics = 5):
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words='english')
    tf = tf_vectorizer.fit_transform(text)

    lda = LatentDirichletAllocation( n_topics=n_topics, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
    lda.fit(tf)

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

def matrix_to_sframe(tfidf):
    '''
    Input: TFIDF numpy sparse matrix
    Output: SFrame with three columns: words, reviews, and tfidf values
    '''
    reviews, words, vals = [], [], []

    for word_idx in range(tfidf.shape[1]):
        word = tfidf.getcol(word_idx).todok()
        keys = word.keys()
        words += [word_idx for key in keys]
        reviews += [key[0] for key in keys]
        vals += word.values()

    return SFrame({'word':words, 'review':reviews, 'tfidf':vals})

def build_recommender(sf, reduced_categories, n_factors=8):
    #Convert reduced_categories to an SFrame
    reduced_review_category = SFrame(pd.DataFrame(reduced_categories))
    #Add the review numbers to serve as indices for the new SFrame
    reduced_review_category.add_column(graphlab.SArray(df.index.get_values()), name='review')

    #Build the recommender
    rec = graphlab.recommender.factorization_recommender.create( sf, user_id='review', item_id='word', target='tfidf', user_data = reduced_review_category, num_factors = n_factors, side_data_factorization = True, nmf=True)
    return rec

def predict_one(rec, datapoint = SFrame({'feature': [4350], 'review': [430800]}) ):
    prediction = "rating:", rec.predict(one_datapoint)[0]
    print(prediction)
    return prediction

def clean_salesRank_col():
    # To clean up and try out

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
    pass

def clean_salesRank_row(row, fill_val = 0):
    # if math.isnan(row) or type(row) is not str:
    #     return fill_val
    # try:
    #     broad_cat = str(row).strip("{}'").split(': ')[0].strip("'")
    #     rank = int(str(row).strip("{}'").split(': ')[1])
    #     return (broad_cat, rank)
    # except TypeError, IndexError:
    #     return (fill_val, fill_val)
    pass

def clustering(tfidf, n_components=100, k=25):
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(tfidf)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    km = MiniBatchKMeans(n_clusters=25, init='k-means++', n_init=1,
                     init_size=1000, batch_size=1000)
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))

def describe_clusters(kmeans_model, df):
    print("Top terms per cluster:")
    clusters = kmeans_model.cluster_info
    cluster_ids = kmeans_model.cluster_id
    print("\n Cluster {} \n".format(centroid_idx))
    for centroid_idx in range(clusters.select_column('cluster_id').max()):
        print("\n Cluster {} \n".format(centroid_idx))
        top_rows = cluster_ids.filter_by(centroid_idx, 'cluster_id').sort('distance').head(5).select_column('row_id')
        for exemplar in top_rows.to_numpy():
            if exemplar < 700000:
                print(df.reviewText[exemplar])
                print()


    # order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    # terms = vectorizer.get_feature_names()
    # for i in range(true_k):
    #     print("Cluster %d:" % i, end='')
    #     for ind in order_centroids[i, :10]:
    #         print(' %s' % terms[ind], end='')
    #     print()

def clean_latent_feature_matrices(model, num_features):
    '''
    Input: Fitted GraphLab recommender model, number of features the model decomposed reviews into
    Output: DataFrame of review latent features from model, DataFrame of word latent features from model
    '''
    # Get dictionary of model coefficients
    model_coefficients = model['coefficients']

    # Turn the review and word coefficients into dataframes
    review_df = model_coefficients['review'].to_dataframe()
    word_df = model_coefficients['word'].to_dataframe()

    # Clean up dataframes by moving latent features from list into separate columns
    review_df, word_df = clean_dfs([review_df, word_df], num_features)

    # Remove unnecessary columns
    review_df, word_df = [df.drop('factors', axis=1) for df in [review_df, word_df]]

    # Reset indices
    review_df.set_index(['review'], inplace=True)
    word_df.set_index(['word'], inplace=True)

    return review_df, word_df

def clean_dfs(dfs, num_features):
    '''
    Input: List of DataFrames with latent features in single list, how many latent features in list
    Output: List of DataFrames with the latent features in seperate columns
    '''
    # For each dataframe make a new column, with rating_* as name, for each latent feature
    return [pd.concat([df] + [pd.Series(df.factors.apply(lambda x: x[i]),
                              name='category_{}'.format(i+1)) for i in range(num_features)],
                      axis=1)  for df in dfs]



def latent_category_examples(df, labels, n_examples = 10):
    '''
    Input: Dataframe containing latent features from factorization recommender, labels for the matrix (either reviews or words), number of examples to return
    Output: Prints n_examples with the highest latent feature scores for each category
    '''

    for col in df.columns:
        top_features = df[col].argsort().values[-n_examples:]
        print('\n Category {} \n'.format(col))
        print(labels[top_features])
        print()

def label_top_category(row):
    if max(row.values) == 0.0:
        return 'None'
    else:
        return np.argmax(row)

def build_model(df, n_latent_categories, vec_type, kwargs):
    #Load the reviews
    reviews = df.reviewText.values

    # Create TFDIF Vectorization as an SFrame
    print("Creating TFIDF Vectorization")
    counts, vectorizer = fit_vectorizer(reviews, kwargs, vec_type)
    #print_topics_nmf(tfidf)
    sf = matrix_to_sframe(counts)
    #sf = graphlab.load_sframe('tfidf_sframe.csv')

    # Load the dimensionality-reduced product categories
    reduced_categories = np.load('reduced_categories.npy')

    print("Building recommender")
    model = build_recommender(sf, reduced_categories, n_latent_categories)

    # Extract latent feature matrices
    review_matrix, word_matrix = clean_latent_feature_matrices(model, n_latent_categories)
    latent_category_examples(review_matrix, reviews, 10)
    words = np.array(vectorizer.get_feature_names())
    latent_category_examples(word_matrix, words, 10)

    return model, review_matrix, word_matrix, vectorizer, counts

if __name__ == '__main__':
    df = pd.read_csv('cleaned_one_star.csv')
    num_latent_categories = 5
    kwargs = {'stop_words':'english', 'tokenizer':None, 'ngram_range': (1,3), 'min_df': 4, 'max_features': 10000,'max_df': 0.95, 'binary':True}
    rec, review_matrix, word_matrix, vectorizer, counts = build_model(df, num_latent_categories, 'count', kwargs)
    rec.save('trigrams_model')

    #top_category = review_matrix.apply(label_top_category, axis=1)

    #subset = review_matrix[top_category == 'category_3']
    #latent_category_examples(subset, reviews, 10)


    # rec.save('model')
    #rec = graphlab.load_model('model')

    # rec.get_similar_users([457612], k=20)

    # recommendations = rec.recommend()
    # # kmeans_rec_model = graphlab.kmeans.create(recommendations, num_clusters=25, max_iterations=2000)
    # # kmeans_rec_model.save('recommendations_kmeans_model')

    # kmeans_model = graphlab.kmeans.create(sf, num_clusters=25, max_iterations=1000)
    # clustering(tfidf)

    # Data pipeline
    # test = df.iloc[0,:]
    # vec = tfidf_vectorizer.transform(test.reviewText)
    # reduced = tsvd_model.transform(vec)
    # rec.predict(reduced)
