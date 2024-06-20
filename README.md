# Suport-Vector-Machine
I use Support Vector Machine and Decision Tree classification models to recognize fraudulent credit card transactions.

I use a real dataset (creditcard.csv, from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) with information about transactions made by credit cards in September 2013 by European cardholders to train each of these models. I then use the trained model to assess if a credit card transaction is legitimate or not. I model the problem as a binary classification problem. A transaction belongs to the positive class (column name 'class') 1 if it is a fraud, otherwise it belongs to the negative class 0.

I use Scikit-Learn and the Python API offered by the Snap Machine Learning (Snap ML) library, a high-performance library for ML modeling. I carry out the following tasks:

Dataset Preprocessing

Dataset Train/Test Split

Build a Decision Tree Classifier model with Scikit-Learn

Build a Decision Tree Classifier model with Snap ML

Evaluate the Scikit-Learn and Snap ML Decision Tree Classifiers (running inference and computing the probabilities of the test samples to belong to the class of fraudulent transactions. I then compute the Area Under the Receiver Operating Characteristic Curve or ROC-AUC score from the predictions)

Build a Support Vector Machine model with Scikit-Learn

Build a Support Vector Machine model with Snap ML

Evaluate the Scikit-Learn and Snap ML Support Vector Machine Models (running inference and getting the confidence scores for the test samples. I then compute the ROC-AUC score from the predictions. I also evaluate the models accuracy by using the hinge loss function and a confussion matrix, for which I also create a function to plot the matrix)


