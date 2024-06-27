# NLP-fake-job-description
The code performs extensive preprocessing on text data, followed by training multiple classification models to detect fraudulent job postings. Each model’s performance is evaluated using standard metrics, providing a comprehensive view of their effectiveness. The neural network, decision tree, naive Bayes, logistic regression, and some other models

Data Preprocessing
Data Loading and Initial Exploration:

The dataset fake_job_postings.csv is loaded, and the first few rows are displayed to understand its structure.
Feature Engineering:

A new column str_information is created by concatenating multiple text columns (title, department, company_profile, description, requirements, benefits, function, required_experience, required_education, industry).
Text Cleaning:

Text is converted to lowercase.
Special characters are removed.
Contractions are expanded using a predefined dictionary.
Text is tokenized, stopwords are removed, and words are lemmatized.
Duplicate Removal and Word Count Calculation:

Duplicate entries in str_information are removed.
The word count for each job posting is calculated and stored in a new column word count.
Tokenization and Padding:

Text is tokenized and converted to sequences.
The sequences are padded to ensure they all have the same length, creating a matrix embedded_docs.
Model Training and Evaluation
Data Preparation:

The target variable y (fraudulent or not) is extracted and reshaped.
Data is split into training and testing sets using train_test_split.
Neural Network Model:

A Sequential neural network model is created with an Embedding layer, Flatten layer, and a Dense layer with sigmoid activation.
Early stopping is employed to prevent overfitting.
The model is trained and evaluated, and its accuracy and loss are displayed.
Other Classifiers:

Decision Tree: A Decision Tree classifier is trained and evaluated.
Naive Bayes: A Gaussian Naive Bayes classifier is trained and evaluated.
Logistic Regression: A Logistic Regression classifier is trained and evaluated.
K-Nearest Neighbors (KNN): A KNN classifier is trained and evaluated.
Random Forest: A Random Forest classifier is trained and evaluated.
Prediction and Evaluation:

A function predict_sample is defined to predict the class of a sample using all trained models.
The classification reports for all models are generated and displayed, showing metrics such as precision, recall, and F1-score.
