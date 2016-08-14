import pandas as pd
import numpy as np
from graphlab import topic_model, SFrame
from graphlab.toolkits.feature_engineering import WordCounter

if __name__ == '__main__':
    df = pd.read_csv('cleaned_one_star.csv')
    text = graphlab.SFrame(df.reviewText)
    word_counter = WordCounter()
    counts = word_counter.fit_transform(text)

    model = topic_model.create(counts, num_topics = 10, num_iterations=25)
    topic_probabilities = model.get_topics(num_words = 10, output_type='topic_words')
