🎬 IMDB Sentiment Analysis using Naive Bayes

This project performs sentiment analysis on IMDB movie reviews using Natural Language Processing (NLP) techniques and Naive Bayes classifiers.
The goal is to classify movie reviews as positive or negative and compare the performance of different Naive Bayes variants.

📌 Dataset

Source: IMDB Movie Reviews Dataset

Total records used: 10,000 (randomly sampled from 50,000)

Columns:

review → Movie review text

sentiment → Positive / Negative (encoded as 1 / 0)

🧹 Text Preprocessing Steps

The following preprocessing steps were applied to clean and prepare the text data:

HTML Tag Removal

Removed tags like <br /> using Regular Expressions.

Lowercasing

Converted all text to lowercase.

Special Character Removal

Retained only alphanumeric characters.

Stopword Removal

Removed common English stopwords using NLTK.

Stemming

Applied Porter Stemmer to reduce words to their root forms.

Tokenization

Converted reviews into lists of meaningful tokens.

🔢 Feature Extraction

Used Bag of Words (BoW) model via CountVectorizer

Converted text into a numerical matrix

Final feature size: 36,187 unique words

X shape: (10000, 36187)
y shape: (10000,)

🔀 Train-Test Split

Training set: 80% (8,000 samples)

Testing set: 20% (2,000 samples)

🤖 Machine Learning Models Used

Three variants of Naive Bayes were implemented and compared:

Gaussian Naive Bayes

Multinomial Naive Bayes

Bernoulli Naive Bayes

These models were trained to find the best classifier for sentiment prediction.

📊 Model Performance (Accuracy)
Model	Accuracy
GaussianNB	63.35%
MultinomialNB	83.70% ✅
BernoulliNB	80.85%
🏆 Conclusion

Multinomial Naive Bayes achieved the highest accuracy

It is best suited for text classification with word frequency features

GaussianNB performed poorly because it assumes normally distributed features, which is not ideal for sparse text data

🛠 Technologies & Libraries Used

Python

NumPy

Pandas

NLTK

Scikit-learn

Regular Expressions (re)

Jupyter Notebook

📁 Project Structure
SentimentAnalysis.ipynb
IMDB.csv
README.md

🚀 Future Improvements

Use TF-IDF Vectorization

Try Logistic Regression / SVM

Apply Lemmatization instead of Stemming

Perform Hyperparameter Tuning

Add Confusion Matrix & F1-score

✍️ Author

Arnab
