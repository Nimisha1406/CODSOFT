"""
Spam SMS Detection System
Author: Nimisha Nandan

"""

#Import Required Libraries

import pandas as pd
import numpy as np
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Text Preprocessing Function

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#Load Dataset

def load_data(path):
    df = pd.read_csv(r"C:\Users\nanda\Downloads\archive\spam.csv", encoding='latin-1')

    # Keep only required columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    print("Dataset Loaded Successfully!")
    print("Total Records:", len(df))
    print(df.head())

    return df

#Preprocess Data

def prepare_features(df):
    
    # Clean messages
    df['message'] = df['message'].apply(preprocess_text)

    X = df['message']
    y = df['label']

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000
    )

    X_tfidf = vectorizer.fit_transform(X)

    return X_tfidf, y, vectorizer

#Train Model

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

#Evaluate Model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nModel Evaluation")
    print("----------------------")
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

#Save Model

def save_model(model, vectorizer):
    with open("spam_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\nModel and Vectorizer saved successfully!")


#Custom Prediction

def predict_message(model, vectorizer):
    """
    Predict custom SMS input
    """
    while True:
        message = input("\nEnter SMS message (type 'exit' to stop): ")

        if message.lower() == 'exit':
            break

        cleaned = preprocess_text(message)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        print("Message Type:", prediction[0])

#Main Function

def main():
    # Load dataset
    df = load_data("spam.csv")

    # Prepare features
    X_tfidf, y, vectorizer = prepare_features(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, vectorizer)

    # Custom prediction
    predict_message(model, vectorizer)


#Run program
if __name__ == "__main__":
    main()
