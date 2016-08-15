import pandas as pd
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
            for i in topic.argsort()[ :-n_top_words - 1:-1]]))
    print()

def print_topics_nmf(tfidf, words, n_topics = 5):
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

def find_topics_by_category(reviews, top_category):
    counts_vectorizers_by_category = {}
    kwargs = {'stop_words':'english', 'tokenizer':LemmaTokenizer(), 'ngram_range': (1,1), 'min_df': 2, 'max_features': 5000,'max_df': 0.95, 'binary':False}

    for cat in set(list(top_category)):
        bool_mask = (top_category == cat)
        try:
            reviews_in_this_category = reviews[bool_mask]
            counts, vectorizer = fit_vectorizer(reviews_in_this_category, kwargs)
            print("\n Category {} \n".format(cat))
            words = vectorizer.get_feature_names()
            nmf = build_optimal_nmf(counts, 10)
            counts_vectorizers_by_category[cat] = (counts, vectorizer, nmf)
            print_top_words(nmf, words, 10)
        except:
            print "No nonzero tfidf features for category {}".format(cat)
    return counts_vectorizers_by_category

if __name__ == '__main__':
    df = pd.read_csv('cleaned_one_star.csv')
    reviews = df.reviewText.values
    reduced_categories = np.load('reduced_categories.npy')
    top_category = np.apply_along_axis(np.argmax, 1, reduced_categories)


    counts_vectorizers_by_category = find_topics_by_category(reviews, top_category)
