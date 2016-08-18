import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD

def save_sparse_matrix(filename,sparse_matrix):
    '''
    Input: Filepath of where to save sparse matrix, Scipy sparse matrix
    Output: Saved sparse matrix to given filepath
    '''
    csr = sparse_matrix.tocsr()
    np.savez(filename, data = csr.data ,indices=csr.indices,
             indptr =csr.indptr, shape=csr.shape )

def load_sparse_matrix(filename):
    '''
    Input: Filename of saved sparse matrix
    Output: Loaded and reconstructed Scipy sparse matrix
    '''
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape']).todok()

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

def print_top_reviews(reduced_categories, num_reviews = 10):
    '''
    Input: Numpy array containing dimensionality-reduced categories,
    integer of number of reviews to be printed
    Output: Prints given number of revies for each of the categories
    '''
    top_reviews_by_cat = np.apply_along_axis(np.argsort, 0, reduced_categories)
    top_10 = top_reviews_by_cat[:num_reviews, :]
    for cat_idx in set(top_category):
        print("\n Category {} \n".format(cat_idx))
        print(reviews[top_10[:, cat_idx]])

def print_reduced_categories(reduced_categories, item_category_mat):
    '''
    Input: Numpy array containing category informed reduced in dimensionality,
    Scipy Sparse Matrix containing the categories for each review
    Output: Prints example reviews for each category
    '''

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

def build_item_category_matrix(categories):
    '''
    Input: Pandas Series containing nested lists of the product categories
    Output: Dictionary of keys sparse matrix with binary labels for each item for each of the nearly 13,000 product categories
    '''

    #Clean category row
    categories = categories.apply(clean_category_row)

    #Initialize a dictionary of keys sparse matrix using a unique, sorted list of the categories for the columns
    category_list = unique(categories)
    item_category_mat = dok_matrix((categories.shape[0], len(category_list)))

    # Build the dictionary of keys sparse matrix from the categories matrix
    for review_idx, review in enumerate(categories):
        for cat in review:
            cat_idx = category_list.index(cat)
            item_category_mat[review_idx, cat_idx] = 1

    return item_category_mat, category_list

def extract_key_categories(item_category_mat, n_categories = 30):
    '''
    Input: Dictionary of keys sparse matrix containing item-category binary values, number of dimensions to which the categories should be reduced
    Output: The Truncated SVD model used to reduce the dimensionality of the sparse matrix, dimensionality-reduced matrix
    '''

    # Build the tsvd model and fit it to the item_category_matrix
    tsvd = TruncatedSVD(n_components= n_categories, n_iter = 50).fit(item_category_mat)

    # Use the tsvd model to reduce the categories matrix down to the number of dimensions specified
    reduced_categories = model.transform(item_category_mat)

    # Return the model and the key_categories matrix
    return model, reduced_categories

def find_optimal_number_of_categories(item_category_mat, low, high, step):
    '''
    Input: Item Category Matrix, the minimum number of categories to consider, the maximum number of categories to consider, and the step size.
    Output: A dictionary in which the keys are the number of categories in the dimensionality-reduced matrix and the value is the percentage of the original variance explained by a dimensionality-reduced matrix with that number categories. Prints percentage of the variance in the categories explained by the number of categories in the range specified.
    '''
    tot_explained_variances = {}
    min_explained_variances = {}
    for n in np.arange(low, high, step):
        model = TruncatedSVD(n_components=n, n_iter = 10)
        model.fit(item_category_mat)
        tot_explained_variances[n] = np.sum( model.explained_variance_ratio_)
        min_explained_variances[n] = np.min( model.explained_variance_ratio_)

    for n in sorted(tot_explained_variances.keys()):
        print(str(n) + " components explained " + str(round(tot_explained_variances[n]*100,2))+ "% of the variance. The marginal component explained " + str(round(min_explained_variances[n]*100,2))+ "% of the variance.")

    return tot_explained_variances, min_explained_variances

def adaptively_find_optimal_number_of_categories(item_category_mat, initial_num_factors, initial_step_size = 5, threshold = .005):
    '''
    Input: Item Category Matrix, the minimum number of categories to consider, the maximum number of categories to consider, and the step size.
    Output: A dictionary in which the keys are the number of categories in the dimensionality-reduced matrix and the value is the percentage of the original variance explained by a dimensionality-reduced matrix with that number categories. Prints percentage of the variance in the categories explained by the number of categories in the range specified

    55 Components was chosen as the optimal value, explaining 55.03% of the variance, with additional components offer little additional explanatory value.
    '''
    tot_explained_variances = {}
    min_explained_variances = {}
    cur_n_factors = initial_num_factors

    # Run model once to initialize
    model = TruncatedSVD(n_components=initial_num_factors, n_iter = 10).fit(item_category_mat)
    tot_explained_variances[initial_num_factors] = np.sum( model.explained_variance_ratio_)
    min_explained_variances[initial_num_factors] = np.min( model.explained_variance_ratio_)

    # Add factors by step size while the marignal factor continues to add explanatory value
    while min_explained_variances[cur_n_factors] > threshold:
        cur_n_factors += initial_step_size

        model = TruncatedSVD(n_components=cur_n_factors, n_iter = 10)
        model.fit(item_category_mat)
        tot_explained_variances[cur_n_factors] = np.sum( model.explained_variance_ratio_)
        min_explained_variances[cur_n_factors] = np.min( model.explained_variance_ratio_)

    # Now trim by one factor at a time to find optimal number of factors
    while min_explained_variances[cur_n_factors] < threshold:
        cur_n_factors -= 1

        model = TruncatedSVD(n_components=cur_n_factors, n_iter = 10)
        model.fit(item_category_mat)
        tot_explained_variances[cur_n_factors] = np.sum( model.explained_variance_ratio_)
        min_explained_variances[cur_n_factors] = np.min( model.explained_variance_ratio_)

    for n in tot_explained_variances.keys():
        print(str(n) + " components explained " + str(round(tot_explained_variances[n]*100,2))+ "% of the variance. The marginal component explained " + str(round(min_explained_variances[n]*100,2))+ "% of the variance.")

    return tot_explained_variances, min_explained_variances, cur_n_factors

if __name__ == "__main__":
    df = pd.read_csv('cleaned_one_star.csv')
    reviews = df.reviewText.values
    item_category_mat, category_list = build_item_category_matrix(df.categories)
    pickle.dump(category_list, open('category_list.pkl', 'wb'))
    #item_category_mat = load_sparse_matrix('item_category_csr.npz')

    total_explained_variance, min_explained_variances, best_n = adaptively_find_optimal_number_of_categories(item_category_mat, 5, 5)
    tsvd = TruncatedSVD(n_components= best_n, n_iter = 50).fit_transform(item_category_mat)
    #pickle.dump(tsvd, open('categories_tsvd.pkl', 'wb'))

    # Use the tsvd model to reduce the categories matrix down to the number of dimensions specified
    reduced_categories = model.transform(item_category_mat)

    #reduced_categories = np.load('reduced_categories_30.npy')
    top_category = np.apply_along_axis(np.argmax, 1, reduced_categories_15)

    # Save reduced categories to file
    np.save('reduced_categories.npy', reduced_categories)
