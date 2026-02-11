
"""
Project Title: Movie Genre Classification using Machine Learning

Author: Nimisha Nandan
"""

#Import Libraries

import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Convert text file to csv file

df=pd.read_csv(r"C:\Users\nanda\Downloads\archive\Genre Classification Dataset\train_data.txt",sep=":::",engine="python",header=None)


df=df[[2,3]]
df.columns=["genre","plot"]

df.to_csv("movie_genre.csv",index=False)

print("Clean CSV created successfully!")
print(df.head())
print(df.shape)

#Text Cleaning Function

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#Load Dataset

def load_data(path):
    df = pd.read_csv("movie_genre.csv")
    print("Dataset Loaded Successfully!")
    print("Total Records:", len(df))
    return df

#Train Models

def train_models(X_train, y_train):

    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    return nb, lr

#Evaluate Model

def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"\n{model_name} Accuracy: {round(acc * 100, 2)}%")
    print("Classification Report:")
    print(classification_report(y_test, predictions))

#Main Function

def main():

    # Load dataset
    df = load_data("movie_genre.csv")

    # Clean text
    df["clean_plot"] = df["plot"].apply(preprocess_text)

    X = df["clean_plot"]
    y = df["genre"]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=7000,
        ngram_range=(1, 2)
    )

    X_features = vectorizer.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train models
    nb_model, lr_model = train_models(X_train, y_train)

    # Evaluate
    evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # Save best model 
    pickle.dump(lr_model, open("genre_model.pkl", "wb"))
    pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

    print("\nModel and Vectorizer saved successfully!")

    # Test with user input
    while True:
        user_input = input("\nEnter a movie plot (type 'exit' to stop): ")

        if user_input.lower() == "exit":
            break

        cleaned = preprocess_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = lr_model.predict(vector)

        print("Predicted Genre:", prediction[0])

#Run Program

if __name__ == "__main__":
    main()
