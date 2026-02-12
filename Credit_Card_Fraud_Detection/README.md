📌 Project Overview
This project implements a Machine Learning based Credit Card Fraud Detection System using Logistic Regression.
The system analyzes transaction data and predicts whether a transaction is fraudulent or legitimate.
Due to the rapid growth of online transactions, detecting fraudulent activity automatically has becomes essential for financial institutions. 
This project demonstrates how supervised machine learning can be applied to solve real-world fraud detection problems.

#Problem Statement

Credit card fraud is a major financial issue worldwide. Detecting fraudulent transactions is challenging because:
-Fraud cases are very rare compared to legitimate transactions (class imbalance problem).
-Fraud patterns continuously evolve.
-Real-time detection is required to prevent losses.

The goal of this project is to build a classification model that can:
-Learn from historical transaction data
-Identify suspicious transactions
-Predict whether a transaction is fraud (1) or legitimate (0)

#Technologies & Libraries Used

-Python
-Pandas – Data handling and manipulation
-NumPy – Numerical operations
-Scikit-learn
-Logistic Regression
-Train-test split
-StandardScaler
-Evaluation metrics (Accuracy, Confusion Matrix, Classification Report)

#Dataset Information

The dataset contains credit card transaction records with:
-Multiple numerical features representing transaction details
-A target column:
    -is_fraud → 1 (Fraud)
    -is_fraud → 0 (Legitimate)
-Example Columns:
   -Transaction amount (amt)
   -Location details (lat, long)
   -Merchant information
   -Timestamp features
   -is_fraud (Target variable)
Dataset link- https://www.kaggle.com/datasets/kartik2112/fraud-detection

#Project Workflow

1️ Data Loading
The dataset is loaded using Pandas.
The system prints:
-Total records
-Fraud case count
-Column names (for verification)

2️ Data Preprocessing
-The target column is_fraud is separated.
-All remaining columns are treated as input features.
-Feature scaling is applied using StandardScaler to normalize data.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
Scaling improves model performance and convergence.

3️ Train-Test Split
The dataset is divided into:
-80% Training data
-20% Testing data  
    train_test_split(test_size=0.2, random_state=42)
    
4️ Model Training
The model used:
-Logistic Regression
    LogisticRegression(max_iter=1000, class_weight='balanced')
    
@Why Logistic Regression?
-Efficient for binary classification
-Works well with scaled numerical data
-Interpretable
-Handles imbalance using class_weight='balanced'

5️ Model Evaluation
The following evaluation metrics are used:
-Accuracy
-Confusion Matrix
-Precision
-Recall
-F1-Score

Example Output:
Accuracy: 0.95
The classification report shows performance for:
-Fraud transactions
-Legitimate transactions

6️ Sample Transaction Prediction
After training, the system tests a sample transaction:
    result = model.predict(sample)
Output example:
Transaction is Legitimate
or
Transaction is Fraudulent

#Model Performance

Accuracy: ~95% (depends on dataset)
Handles class imbalance using class_weight='balanced'
Evaluated using confusion matrix and classification report

~ Note: In highly imbalanced datasets, accuracy alone is not sufficient. Precision and recall are important metrics.

#How to Run the Project

Step 1: Install Required Libraries
pip install pandas numpy scikit-learn

Step 2: Place Dataset
Ensure the csv file is in the same folder as the Python file.

Step 3: Run the Script
    python Credit card fraud detection.py
    
The program will:
-Load dataset
-Train model
-Evaluate performance
-Show sample transaction prediction
