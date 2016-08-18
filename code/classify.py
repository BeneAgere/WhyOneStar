import pickle

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.ret = RegexpTokenizer('\w+')
    def __call__(self, doc):
        tokens = [self.wnl.lemmatize(t) for t in self.ret.tokenize(doc)]
        return [t for t in tokens if len(t) > 1 and not is_number(t)]

class ComplaintClassifier(object):


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

def classify(text, categories):
    cat = top_category(categories, category_list, categories_tsvd)
    tfidf, vectorizer, nmf = counts_vectorizers_by_category[29]
    prediction_tfidf = vectorizer.transform(text)
    nmf.transform(vectorizer.transform(text))

if __name__ == '__main__':
    item_category_mat = load_sparse_matrix('./item_category_csr.npz')
    reduced_categories = np.load('reduced_categories_30.npy')
    counts_vectorizers_by_category = pickle.load(open('vectorization_by_category.pkl', 'rb'))
    categories_tsvd = pickle.load(open('categories_tsvd.pkl', 'rb'))
    category_list = pickle.load(open('category_list.pkl', 'rb'))
