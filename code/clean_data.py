import pandas as pd
import json
import gzip

def load_json(filepath):
    '''
    Input: Filepath of JSON file
    Output Dataframe containing JSON data
    '''
    data = read_file(filepath)
    clean_data = remove_problem_lines(data)
    return read_json(data)

def read_file(filepath):
    '''
    Input: Filepath
    Output: Lines read from file
    '''
    with open(filepath) as f:
        data = f.readlines()
    return data

def remove_problem_lines(data, error_rate = False):
    '''
    Iterates through the list of json entries passed in and identifies the indices for each of the entries that fail to load. The function then removes these lines from the list of values. If error rate is set to true, prints the error rate.
    '''
    problem_lines = []
    num_errors = 0
    for index, line in enumerate(data):
        try:
            line = line.strip()
            json.loads(line)
        except ValueError:
            problem_lines.append(index)
            num_errors += 1
            print("Error reading line: {}".format(index))

    if error_rate:
        print("Percent errors: {}".format(len(problem_lines)/float(len(data))))

    for index in problem_lines:
        data.pop(index)
    return data

def read_json(json_file):
    '''
    Input: JSON file
    Output: Pandas DataFrame containing values from JSON file
    '''
    json_file = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ",".join(json_file) + "]"
    return pd.read_json(data_json_str)

def parse_gzip(path):
    '''
    Helper function for getDF
    Input: filepath
    Output: A generator containing the lines in the given file.
    '''
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    '''
    Input: Filepath of gzip containing a dataframe
    Output: The unzipped dataframe
    '''
    i = 0
    df = {}
    for d in parse_gzip(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def remove_nan_reviews(data):
    '''
    Input: Numpy array of reviews
    Output: Numpy array of reviews with all blank entries removed
    '''
    nan_indices = []
    lengths = []
    for index, line in enumerate(data):
        try:
            lengths.append(len(line))
        except TypeError:
            nan_indices.append(index)
    print("{} reviews were empty and not included.".format(len(nan_indices)))
    return np.delete(data, nan_indices), nan_indices

if __name__ == '__main__':
    '''
    Read review and metadata json files to Pandas DataFrames, then export to csv for ease of access.
    '''
    df = load_json('reviews.json')
    one_star = df[df.overall == 1]

    # Drop entires that lack reviews
    reviews, nan_indices = remove_nan_reviews( df.reviewText.values)
    df = df.drop(df.index[[nan_indices]])

    metadata = getDF('metadata.json')
    df = pd.merge(one_star, metadata, on='asin')
    df.to_csv('cleaned_one_star.csv')
