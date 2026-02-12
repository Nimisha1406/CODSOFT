📌 Project Overview

This project implements a Spam SMS Detection System using Machine Learning and Natural Language Processing (NLP) techniques.
The system classifies SMS messages into:
-Spam (unwanted promotional or fraudulent messages)
-Ham (legitimate messages)
The model is trained using TF-IDF vectorization and Logistic Regression to automatically detect spam messages with high accuracy.

#Problem Statement
Spam messages are a common issue in mobile communication. They can include:
-Fake lottery messages
-Fraudulent banking alerts
-Promotional advertisements
-Phishing attempts
The objective of this project is to build a classification model that:
-Learns from labeled SMS data
-Converts text into numerical features
-Predicts whether a message is spam or ham

#Technologies & Libraries Used

-Python
-Pandas – Data handling
-NumPy – Numerical operations
-Scikit-learn
-TF-IDF Vectorizer
-Logistic Regression
-Train-Test Split
-Accuracy & Classification Report
-Regular Expressions (re) – Text cleaning
-Pickle – Model saving

#Dataset Information

File name: spam.csv
Dataset link- https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
Encoding: latin-1
Columns used:
-v1 → Label (spam / ham)
-v2 → Message text
After preprocessing:
-label → Target variable
-message → Input feature
The dataset contains thousands of SMS messages labeled as spam or ham.

#Project Workflow
1️ Data Loading
The dataset is loaded and unnecessary columns are removed.
    df = pd.read_csv(r"C:\Users\nanda\Downloads\archive\spam.csv", encoding='latin-1')
Only relevant columns are kept:
-label
-message

2️ Text Preprocessing
Each message undergoes cleaning:
-Convert text to lowercase
-Remove special characters
-Remove extra spaces
Example:
Before:
Congratulations!!! You WON $1000!!!
After:
congratulations you won

3️ Feature Extraction (TF-IDF)
Text messages are converted into numerical feature vectors using:
    TfidfVectorizer(stop_words='english', max_features=3000)
TF-IDF (Term Frequency – Inverse Document Frequency):
-Gives importance to meaningful words
-Reduces impact of common words like "the", "is", etc.

4️ Train-Test Split
The dataset is divided into:
-80% Training Data
-20% Testing Data
This ensures proper evaluation of model performance.

5️ Model Training
Model Used:
-Logistic Regression
-LogisticRegression(max_iter=1000)

@Why Logistic Regression?
-Suitable for binary classification
-Fast and efficient
-Works well with text data
-Provides high accuracy

6️ Model Evaluation
Model performance is evaluated using:
-Accuracy Score
-Precision
-Recall
-F1-Score
-Classification Report

Expected Accuracy:
  ~96% to 98%
  
7️ Model Saving
The trained model and TF-IDF vectorizer are saved using Pickle:
-spam_model.pkl
-tfidf_vectorizer.pkl
This allows reuse without retraining.

8️ Custom SMS Prediction
The system allows user input:
Enter SMS message (type 'exit' to stop):
Example:
Input:
Congratulations! You have won a free gift voucher!
Output:
Message Type: spam

#Model Performance

Typical results:
-Accuracy: ~97%
-High precision for spam detection
-Balanced performance between spam and ham

The model performs well due to effective text preprocessing and TF-IDF feature extraction.

#How to Run the Project

Step 1: Install Required Libraries
    pip install pandas numpy scikit-learn
    
Step 2: Place Dataset
Ensure spam.csv is in the same directory as the Python file.

Step 3: Run the Script
    python spam_detection.py
    
Step 4: Enter SMS
Type a message to test detection.

Type exit to stop the program.
