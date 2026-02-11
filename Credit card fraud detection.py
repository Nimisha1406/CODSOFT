
"""
Project Title: Credit Card Fraud Detection

"""
#Import Libraries

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

#Load Dataset

def load_data(path):
    df = pd.read_csv(r"C:\Users\nanda\Downloads\archive (1)\fraudTrain.csv")
    print(df.columns)
    print("Dataset Loaded Successfully!")
    print("Total Records:", len(df))
    print("Fraud Cases:", df['is_fraud'].sum())
    return df

#Preprocess Data

def preprocess_data(df):
    numeric_df=df.select_dtypes(include=['int64','float64'])
    X = numeric_df.drop("is_fraud", axis=1)
    y = numeric_df["is_fraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

#Train Model

def train_model(X_train, y_train):

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    return model

#Evaluate Model

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("\nModel Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


#Main Function

def main():

    # Load data
    df = load_data("creditcard.csv")

    # Preprocess
    X_scaled, y, scaler = preprocess_data(df)

    # Train-test split (stratified because dataset is imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save model and scaler
    pickle.dump(model, open("fraud_model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))

    print("\nModel and Scaler saved successfully!")

    # Test with one random transaction
    sample = X_test[0].reshape(1, -1)
    prediction = model.predict(sample)

    if prediction[0] == 1:
        print("\nTransaction is Fraudulent")
    else:
        print("\nTransaction is Legitimate")


#Run Program

if __name__ == "__main__":
    main()
