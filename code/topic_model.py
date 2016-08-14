import pandas as pd
import numpy as np
import graphlab
from graphlab import topic_model, SFrame
from graphlab.toolkits.feature_engineering import WordCounter
from sklearn.feature_extraction.text import CountVectorizer

def to_sarray(dok, words):
    dictionaries = []
    for row_idx, row in enumerate(dok):
        row_dictionary = {}
        for key in row.iterkeys():
            row_dictionary[words[key[1]]] = row[key]
        dictionaries.append(row_dictionary)

    return graphlab.SArray(dictionaries)

def graphlab_counts(text):
    words = ['fit', 'broken', 'return', 'late', 'packaging', 'big', 'small', 'broke', 'broken', 'boring', 'expensive', 'cheap', 'flimsy', 'waste']
    # stopwords = list(graphlab.text_analytics.stopwords())
    # stopwords += ['the', 'i', 'to', 'this', 'was', 'that', 'my', 'for', 'but', 'and', 'in', 'the', 'is', 'it', 'too', 'them', 'a', 'of', 'these', 'you', 'was', 'after']
    word_counter = WordCounter(delimiters=None)
    return word_counter.fit_transform(text)

def sklearn_wordcounts(text):
    text = list(text)
    cv = CountVectorizer(stop_words='english', min_df = 3, max_df = 0.9, max_features = 10000)
    words = cv.get_feature_names()
    #counts = cv.fit_transform(text)

    return counts_dictionary = to_sarray(counts.todok(), words)

def print_results(model):
    topic_words = model.get_topics(num_words = 10, output_type='topic_words')
    for row in topic_words:
        print(row['words'])

    topic_probabilities = model.get_topics(num_words = 10, output_type='topic_probabilities')
    topic_probabilities.print_rows(num_rows=100)
    pass

if __name__ == '__main__':
    df = pd.read_csv('cleaned_one_star.csv')

    counts = graphlab_counts(raphlab.SFrame(df.reviewText))
    #counts = sklearn_wordcounts(df.reviewText.values)

    model = topic_model.create(counts, num_topics = 10, num_iterations=25, alpha=0.6, beta=0.2)
    print_results(model)
