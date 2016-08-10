import pandas as pd
import json
import gzip

def load_json(filepath):
    data = read_file(filepath)
    clean_data = remove_problem_lines(data)
    return read_json(data)

def read_file(filepath):
    with open(filepath) as f:
        data = f.readlines()
    return data

def remove_problem_lines(data, error_rate = False):
    '''
    Iterates through the list of json entries passed in and identifies the indices for each of the entries that fail to load. The function then removes these lines from the list of values. If error rate is set to true, prints the error rate.
    '''
    problem_lines = []
    for index, line in enumerate(data):
        try:
            line = line.strip()
            json.loads(line)
        except ValueError:
            problem_lines.append((index, line))
            num_errors += 1
            print("Error reading line: {}".format(index))

    if error_rate:
        print("Percent errors: {}".format(len(problem_lines)/float(len(data))))

    for index in problem_lines(data):
        data.pop(index)
    return data

def read_json(data):
    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ",".join(data) + "]"
    return pd.read_json(data_json_str)

def parse_gzip(path):
    g = gzip.open(path, 'rb')
    for l in g:
    yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse_gzip(path):
    df[i] = d
    i += 1
    return pd.DataFrame.from_dict(df, orient='index')

features = {}
for review_idx, review_vector in enumerate(S.keys()):
    features[review_idx] = SArray(review_vector)


if __name__ == '__main__':
    '''
    Read review and metadata json files to Pandas DataFrames, then export to csv for ease of access.
    '''
    data = load_json('reviews.json')
    one_star = df[df.overall == 1]
    data.to_csv('one_star.csv')

    metadata = getDF('metadata.json')
    metadata.to_csv('metadata.csv')
