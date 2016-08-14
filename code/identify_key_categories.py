import pandas as pd
from scipy.sparse import dok_matrix
from sklearn.decomposition import TruncatedSVD

def unique(series):
    '''
    Input: Pandas series
    Output: Sorted list with the unique objects in the series.
    '''
    collection = []
    for row in series:
        for item in row:
            collection.append(item)
    return sorted(list(set(collection)))

def print_reduced_categories(reduced_categories, item_category_mat):

    for red_cat_idx in range(reduced_categories.shape[1]):
        rep_indices = reduced_categories[:,red_cat_idx ].argsort()[-10:]
        exemplar_subset_mat = item_category_mat[rep_indices]

        print("\nReduced Category Number {}\n".format(red_cat_idx))
        for review in exemplar_subset_mat:
            keys = review.keys()
            indices = [x[1] for x in keys]
            cats = []
            for i in indices:
                cats.append(category_list[i])
            print(cats)

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

def extract_key_categories(categories, n_categories = 55):
    '''
    Input: Pandas Series containing nested lists of the product categories, number of dimensions to which the categories should be reduced
    Output: The Truncated SVD model used to reduce the dimensionality of the sparse matrix
    '''

    #Clean category row
    categories = categories.apply(clean_category_row)

    #Initialize a dictionary of keys sparse matrix using a unique, sorted list of the categories for the columns
    category_list = unique(categories)
    item_category_mat = dok_matrix((categories.shape[0], len(category_list)))

    # Build the dictionary of keys sparse matrix from the categories matrix
    for review_idx, review in enumerate(cats):
        for cat in review:
            cat_idx = category_list.index(cat)
            item_category_mat[review_idx, cat_idx] = 1

    # Build the tsvd model and fit it to the item_category_matrix
    tsvd = TruncatedSVD(n_components= n_categories, n_iter = 100).fit(item_category_mat)

    # Use the tsvd model to reduce the categories matrix down to the number of dimensions specified
    key_categories = model.transform(item_category_mat)

    # Return the model and the key_categories matrix
    return model, key_categories

def find_optimal_number_of_categories(item_category_mat, low, high, step):
    '''
    Input: Item Category Matrix, the minimum number of categories to consider, the maximum number of categories to consider, and the step size.
    Output: A dictionary in which the keys are the number of categories in the dimensionality-reduced matrix and the value is the percentage of the original variance explained by a dimensionality-reduced matrix with that number categories. Prints percentage of the variance in the categories explained by the number of categories in the range specified

    55 Components was chosen as the optimal value, explaining 55.03% of the variance, with additional components offer little additional explanatory value.
    '''
    explained_variances = {}
    for n in np.arange(low, high, step):
        model = TruncatedSVD(n_components=n, n_iter = 10)
        model.fit(item_category_mat)
        print(model.explained_variance_ratio_)
        explained_variances[n] = model.explained_variance_ratio_


    for n in sorted(explained_variances.keys()):
        print str(n) + " components explained " + str(round(np.sum(explained_variances[n])*100,2))+ "% of the variance"
    return explained_variances

if __name__ == "__main__":
    df = pd.read_csv('cleaned_one_star.csv')
    print("Reducing category features")
    tsvd_model, reduced_categories = extract_key_categories(df.categories)

    # Save reduced categories to file
    np.save('reduced_categories', reduced_categories)
