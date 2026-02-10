import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("movie_genre.csv")

print("Dataset shape:", data.shape)
print(data.head())

# Text and labels
plot_text = data['plot']
genre_table= data['genre']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)
X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Testing with custom input
sample_plot = ["A young boy discovers he has magical powers and saves the world"]
sample_vec = tfidf.transform(sample_plot)
prediction = model.predict(sample_vec)

print("Predicted Genre:", prediction[0])
