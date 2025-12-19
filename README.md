# IMDB Sentiment Analysis using Naive Bayes 🎬📝

This project performs **sentiment analysis** on IMDB movie reviews using **Natural Language Processing (NLP)** techniques and **Naive Bayes classifiers**. The goal is to classify movie reviews as **positive** or **negative** and compare the performance of different Naive Bayes variants.

---

## 📂 Dataset

* **File name**: `IMDB.csv`
* **Description**: Contains 50,000 movie reviews with sentiment labels (positive/negative).
* **Sample used**: 10,000 randomly sampled reviews for faster computation.
* ⚠️ Ensure `IMDB.csv` is present in the project root directory before running the code.

---

## 🛠️ Technologies & Libraries Used

* Python 3.x
* NumPy – Numerical computations
* Pandas – Data manipulation
* NLTK – Natural Language Processing
* Scikit-learn – Machine learning algorithms

```bash
pip install numpy pandas nltk scikit-learn
```

---

## 🔍 Workflow Overview

### 1️⃣ Data Loading & Inspection

* Load CSV data using Pandas
* Inspect dataset shape and head
* Check for null values
* View summary statistics

### 2️⃣ Data Cleaning

* Remove HTML tags using Regular Expressions
* Convert text to lowercase
* Remove special characters
* Remove stopwords using NLTK
* Apply stemming using Porter Stemmer

### 3️⃣ Feature Extraction

* Convert text into numerical features using **Bag of Words** (`CountVectorizer`)
* Final feature matrix shape: `(10000, 36187)`
* Labels converted to 0 (negative) and 1 (positive)

### 4️⃣ Train-Test Split

* Split data into **training (80%)** and **testing (20%)** sets
* Shapes:

  * `X_train`: (8000, 36187)
  * `X_test`: (2000, 36187)
* `y_train`: (8000,)
* `y_test`: (2000,)

### 5️⃣ Model Training

* Implemented three variants of **Naive Bayes**:

  1. GaussianNB
  2. MultinomialNB
  3. BernoulliNB
* Fit each model on the training data

### 6️⃣ Model Evaluation

* Predicted sentiments for test data
* Calculated accuracy for each model

---

## 📊 Model Performance (Accuracy)

| Model         | Accuracy |
| ------------- | -------- |
| GaussianNB    | 63.35%   |
| MultinomialNB | 83.70% ✅ |
| BernoulliNB   | 80.85%   |

---

## 📌 Conclusion

* **Multinomial Naive Bayes** performed the best for text classification with word frequency features.
* GaussianNB is less effective because it assumes normally distributed features, which is not ideal for sparse text data.
* BernoulliNB works well for binary features but slightly lower than MultinomialNB.

---

## 🛠️ How to Run

```bash
python main.py
```

> Or run the notebook cell-by-cell if using **Jupyter Notebook**.

---

## 📁 Project Structure

```
├── IMDB.csv
├── main.py
├── README.md
```

---

## ⭐ Key Takeaways

* Proper **text preprocessing** improves model performance.
* **Bag of Words** and **Naive Bayes** are effective for sentiment classification.
* Model choice matters depending on the **feature distribution**.

---

## 📜 License

This project is for **educational purposes only**.

---

## 🚀 Happy Learning & Sentiment Analysis!
