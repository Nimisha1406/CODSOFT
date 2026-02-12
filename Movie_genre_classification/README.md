📌 Project Description
This project is a Machine Learning based text classification system that predicts the genre of a movie based on its plot description.
It uses Natural Language Processing (NLP) techniques and machine learning algorithms to analyze text and classify movies into their respective genres automatically.

#Problem Statement:

With thousands of movies released every year, manually categorizing them into genres is time-consuming.
This project aims to build an automated system that can predict the genre of a movie using its storyline or description.
Such systems are useful in:
-Movie recommendation systems
-Streaming platforms
-Content filtering and organization

#Technologies Used:

-Python
-Pandas
-NumPy
-Scikit-learn
-Natural Language Processing (NLP)
-TF-IDF Vectorization

#Dataset Information:

Dataset: Genre Classification Dataset IMDb
Dataset Link: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb
Format: Text dataset containing movie plot and corresponding genre
Total Records: 54,214
Each record contains:
Genre (Target variable)
Plot description (Input feature)

#Project Workflow:

1 Data Loading
The dataset is loaded and converted into a clean CSV format.

2️ Text Preprocessing
Converted text to lowercase
Removed special characters
Removed extra spaces

3️ Feature Extraction
TF-IDF (Term Frequency – Inverse Document Frequency) is used to convert text into numerical feature vectors.

4️ Train-Test Split
The dataset is divided into:
80% Training data
20% Testing data

5️ Model Training
Two machine learning models were implemented:
-Multinomial Naive Bayes
-Logistic Regression

6️ Model Evaluation
Model performance was evaluated using:
Accuracy Score
Classification Report (Precision, Recall, F1-score)

#Model Performance:
Naive Bayes Accuracy: ~52%
Logistic Regression Accuracy: ~58%
Logistic Regression performed better compared to Naive Bayes for this dataset.

#Model Saving:
The trained model and TF-IDF vectorizer were saved for future predictions without retraining.

#How to Run:
1️ Install required libraries:
        pip install pandas numpy scikit-learn
2️ Place the dataset file in the project directory
3️ Run the Python script:
        python movie_genre.py
4️ Enter a movie plot description when prompted
5️ The system will predict the movie genre
Type exit to stop the program.

#Sample input:
A young boy discovers he has magical powers and must save the world.
#Output:
Predicted Genre: Adventure
