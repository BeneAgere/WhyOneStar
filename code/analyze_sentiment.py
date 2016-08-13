from textblob import TextBlob
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

def analyze_sentiment(text):
    polarity_scores = []
    subjectivity_scores = text
    for review in text:
        tb = TextBlob(review)
        polarity_scores.append(tb.sentiment.polarity)
        subjectivity_scores.append(tb.sentiment.subjectivity)
    return polarity_scores, subjectivity_scores

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
