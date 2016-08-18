import pandas as pd
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
import string
import sys
import pickle

class LemmaTokenizer(object):
    '''
    Custom Tokenizer that uses WordNet's Lemmatizer and limits to words of at least two chracters
    '''
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.ret = RegexpTokenizer('\w+')
    def __call__(self, doc):
        tokens = [self.wnl.lemmatize(t) for t in self.ret.tokenize(doc)]
        return [t for t in tokens if len(t) > 1 and not is_number(t)]

def is_number(string):
    '''
    Input: String
    Output: Boolean value, True if string can be cast as a float, False otherswise
    '''
    try:
        float(string)
        return True
    except ValueError:
        return False

def print_top_words(model, feature_names, n_top_words):
    '''
    Input: Model, list of words used in vectorization, int number of words to be printed
    Output: Prints given number of words for each model componennt
    '''
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
            for i in topic.argsort()[ :-n_top_words - 1:-1]]))

def print_topics_nmf(tfidf, words, n_topics = 5):
    '''
    Input: TFIDF counts, words list used in the vectorization, and number of Topics
    Output: Prints the top words in each of the specified number of latent features
    '''
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    print("\nTopics in NMF model:")
    print_top_words(nmf, words, n_top_words=10)
    pass

def build_optimal_nmf(tfidf, max_components = 10):
    '''
    Input: Numpy array with tfidf values, maximum number of components to check
    Output: NMF object with the optimal number of components

    Method: Calculates NMF factorizations until it finds the final additional factor that reduces reconstruction error by at least 1 percent from the original.
    '''
    factorizations, errors = [], []
    for x in range(2, max_components+1):
        nmf = NMF(n_components=x, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
        print("With {} factors, reconstruction error is {}".format(x, nmf.reconstruction_err_))
        factorizations.append(nmf)
        errors.append(nmf.reconstruction_err_)

        # Once at least two NMFs have been constructed, check to see if the additional factor reduces the reconstruction error. If not, return the best model.
        if (x > 3) and (((errors[-2] - errors[-1])/errors[0]) < 0.01):
            print("Optimal NMF with {} factors chosen.".format(x-1))
            return factorizations[-2]
    return factorizations[-1]

def fit_vectorizer(text, kwargs, type='tfidf'):
    '''
    Input: Numpy array of review text, dictionary of keyword arguments, and
    string containing the thype of vectorizer to fit.
    Output: Vectorized counts for the given text, vectorizer
    '''
    if type == 'count':
        vectorizer = CountVectorizer(**kwargs)
        counts = vectorizer.fit_transform(text)
    else:
        vectorizer = TfidfVectorizer(**kwargs)
        counts = vectorizer.fit_transform(text)
    return counts, vectorizer

def find_latent_features_by_category(reviews, top_category):
    '''
    Input: Numpy array containing review text, numpy array containing list of category labels
    Output: Dictionary in which the keys are the category labels and the values are tuples containing the tfidf values, the tfidf vectorizer, and the NMF model that has been fit to the tfidf values.
    '''

    counts_vectorizers_by_category = {}
    kwargs = {'stop_words':'english', 'tokenizer':LemmaTokenizer(), 'ngram_range': (1,1), 'min_df': 3, 'max_features': 5000,'max_df': 0.9, 'binary':False}

    for cat in sorted(set(list(top_category))):
        bool_mask = (top_category == cat)
        try:
            reviews_in_this_category = reviews[bool_mask]
            counts, vectorizer = fit_vectorizer(reviews_in_this_category, kwargs)
            print("\n Category {} \n".format(cat))
            nmf = build_optimal_nmf(counts, 10)
            counts_vectorizers_by_category[cat] = (counts, vectorizer, nmf)
            words = vectorizer.get_feature_names()
            print_top_words(nmf, words, 10)
        except IOError as e:
            print "\n I/O error({0}): {1}".format(e.errno, e.strerror)
        except ValueError:
            print "\n Value error on category {}.".format(cat)
        except NameError:
            print "\n Name error on category {}.".format(cat)
    return counts_vectorizers_by_category

'''
Loads the data, assigns reviews to clusters, performs TFIDF vectorization and
NMF within each of the clusters, then saves a pickle file containing the
TFIDF values, vectorizer, and NMF for each cluster.
'''
if __name__ == '__main__':
    df = pd.read_csv('cleaned_one_star.csv')
    reviews = df.reviewText.values
    reduced_categories = np.load('reduced_categories_30.npy')
    top_category = np.apply_along_axis(np.argmax, 1, reduced_categories)
    counts_vectorizers_by_category = find_latent_features_by_category(reviews, top_category)
    pickle.dump(counts_vectorizers_by_category, open('vectorization_by_category.pkl', 'wb'))
