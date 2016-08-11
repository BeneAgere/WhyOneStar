# investihate
Galvanize Data Science Immersive Capstone

Curious what signal there might be in the deafening noise of angry customers? InvestiHate identifies latent features from hundreds of thousands of customer complaints, then categorizes new reviews according to these categories, in addition to providing sentiment analysis, subjectivity measures, and comparisons to other products within the same category. Understanding what does (and doesn't!) make your customers happy has never been so easy.

Technical Details:
Built using a GraphLab Factorization Recommender on the TFIDF featurization of review text. Categories are controlled using side data factorization on the category type, which was identified through Truncated SVD perform dimensionality reduction from 13,000+ categories on Amazon to 55 category types. Sentiment and subjectivity analysis were performed using TextBlob.
