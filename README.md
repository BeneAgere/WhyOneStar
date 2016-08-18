# WhyOneStar?
Galvanize Data Science Immersive Capstone

Curious what your critical customers really think? WhyOneStar quickly and accurately identifies latent features from hundreds of thousands of customer complaints, then assigns new reviews according to these categories. The platform then also provides sentiment analysis and subjectivity measures. Understanding what does (and doesn't!) make your customers happy has never been so easy.

Technical Details:
Built using dimensionality reduction and clustering on 13,000 unique product categories, then performing separate TF-IDF featurization and non-negative matrix factorization to extract latent features within each product category cluster across 700,000 total reviews. All processing was done using an EC2 AWS instance. Words were processed using the WordNet Lemmatizer, and sentiment and subjectivity analysis were performed using TextBlob.
