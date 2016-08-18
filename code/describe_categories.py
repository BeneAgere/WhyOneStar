import pickle
from nltk.sentiment import sentiment_analyzer

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.ret = RegexpTokenizer('\w+')
    def __call__(self, doc):
        tokens = [self.wnl.lemmatize(t) for t in self.ret.tokenize(doc)]
        return [t for t in tokens if len(t) > 1 and not is_number(t)]


def load_sparse_matrix(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape']).todok()

def clean_category_row(row):
    '''
    Input: A single row of categories
    Output: A list of categories with all unnecessary whitespace, nested lists, and quotation marks removed .
    '''
    items = str(row).strip('[]').split(",")
    cleaned_items = []
    for item in items:
        temp = item.strip("' [],'")
        cleaned_items.append(temp)
    return cleaned_items

def top_category(categories, category_list, categories_tsvd):
    '''
    Input: String containing the
    Output: int of the main category number from the categories provided
    '''
    #ret = RegexpTokenizer('\w+')
    #clean_categories = ret.tokenize(categories)
    clean = clean_category_row(categories)

    prediction_categories = dok_matrix((1, 12944))
    for item in clean_test_cats:
        try:
            cat_idx = category_list.index(item)
        except ValueError:
            cat_idx = 0
        prediction_categories[0, cat_idx] = 1

    # Return the index of the top category once transformed to the dimensionality-reduced space
    return np.argmax(categories_tsvd.transform(prediction_categories))

from textblob import TextBlob
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

def analyze_sentiment(text):
    polarity_scores = []
    subjectivity_scores = []
    for review in text:
        tb = TextBlob(review)
        polarity_scores.append(tb.sentiment.polarity)
        subjectivity_scores.append(tb.sentiment.subjectivity)
    return np.mean(polarity_scores), np.mean(subjectivity_scores)

def vader(df):
    sentiments = []
    #one_star.reviewText.apply(nltk.sentiment.util.demo_liu_hu_lexicon)
    #nltk.sentiment.util.demo_liu_hu_lexicon(df.reviewText.iloc[0])

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

def analyze_sentiment(text):
    polarity_scores = []
    subjectivity_scores = text
    for review in text:
        tb = TextBlob(review)
        polarity_scores.append(tb.sentiment.polarity)
        subjectivity_scores.append(tb.sentiment.subjectivity)
    return polarity_scores, subjectivity_scores

def describe_categories():
    counts_vectorizers_by_category = {}

    for cat in sorted(set(list(top_category))):
        bool_mask = (top_category == cat)
        reviews_in_this_category = reviews[bool_mask]


    return counts_vectorizers_by_category


if __name__ == '__main__':
    df = pd.read_csv('cleaned_one_star.csv')
    reviews = df.reviewText.values

    item_category_mat = load_sparse_matrix('./item_category_csr.npz')
    reduced_categories = np.load('reduced_categories_30.npy')
    counts_vectorizers_by_category = pickle.load(open('vectorization_by_category.pkl', 'rb'))
    categories_tsvd = pickle.load(open('categories_tsvd.pkl', 'rb'))
    category_list = pickle.load(open('category_list.pkl', 'rb'))
    top_category = np.apply_along_axis(np.argmax, 1, reduced_categories)
    describe_categories()
