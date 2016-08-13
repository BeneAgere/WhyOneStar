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

def fit_tfidf(text):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2, max_features = 5000)
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
    print prediction
    return prediction

def extract_side_features():
    side_features = has_reviews[['reviewTime', 'categories', 'price', 'salesRank']]
    for col in side_features.columns:
        side_features[col] = str()
    review_data = SFrame(side_features)

    side_features = has_reviews[['reviewTime', 'categories', 'price', 'salesRank']]
    return side_features

def extract_key_categories(categories, n_categories = 55):
    categories = categories.apply(clean_category_row)
    category_list = unique(categories)
    item_category_mat = dok_matrix((categories.shape[0], len(category_list)))

    for review_idx, review in enumerate(categories):
        for cat in review:
            cat_idx = category_list.index(cat)
            item_category_mat[review_idx, cat_idx] = 1

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

    model = TruncatedSVD(n_components= n_categories, n_iter = 30).fit(item_category_mat)
    return model, model.transform(item_category_mat)

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

def unique(series):
    '''
    Returns a sorted list with the unique categories in the series.
    '''
    collection = []
    for row in series:
        for item in row:
            collection.append(item)
    return sorted(list(set(collection)))

def clean_category_row(row):
    '''
    When passed a single row of categories, removes all unnecessary whitespace, nested loops, and additional quotation marks and returns a single list of categories.
    '''
    items = str(row).strip('[]').split(",")
    cleaned_items = []
    for item in items:
        temp = item.strip("' [],'")
        cleaned_items.append(temp)
    return cleaned_items

def label_reduced_categories(reduced_categories, item_category_mat):

    for red_cat_idx in range(reduced_categories.shape[1]):
        rep_indices = reduced_categories[:,red_cat_idx ].argsort()[-10:]
        exemplar_subset_mat = item_category_mat[rep_indices]

        print "\nReduced Category Number {}\n".format(red_cat_idx)
        for review in exemplar_subset_mat:
            #look_at_one = rep_review_category_mat[0]
            keys = review.keys()
            indices = [x[1] for x in keys]
            cats = []
            for i in indices:
                cats.append(category_list[i])
            print(cats)

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
                print df.reviewText[exemplar]
                print


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


# def example_reviews_for_latent_categories(review_matrix):
#     for col in review_matrix.columns:
#         a = review_matrix['category_1']
#         b = np.argsort(a.values)[-10:]


if __name__ == '__main__':
    df = pd.read_csv('cleaned_one_star.csv')
    #reviews = pd.read_csv('cleaned_one_star.csv').reviewText.values
    #tfidf, tfidf_vectorizer = fit_tfidf(reviews)
    #print_topics_nmf(tfidf)
    tsvd_model, reduced_categories = extract_key_categories(df.categories)

    #sf = matrix_to_sframe(tfidf)
    sf = graphlab.load_sframe('tfidf_sframe.csv')
    rec = build_recommender(sf, reduced_categories, 12)
    rec.save('model')

    #rec = graphlab.load_model('model')
    factor_matrix, review_matrix = clean_latent_feature_matrices(rec, num_features = 12)



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
