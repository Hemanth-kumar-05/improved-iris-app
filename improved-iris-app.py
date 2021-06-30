# S2.1: Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'improved_iris_app.py'.
# Importing the necessary libraries.
from re import I
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating an SVC model. 
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)

# Creating a Logistic Regression model. 
rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100)
rf_clf.fit(X_train, y_train)

# Creating a Random Forest Classifier model.
log_reg = LogisticRegression(n_jobs = -1)
log_reg.fit(X_train, y_train)

@st.cache()
def prediction(model, sepal_length, sepal_width, petal_length, petal_width):
    pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = pred[0]
    if pred == 0:
        return "Iris-setosa"
    elif pred == 1:
        return "Iris-virginica"
    else:
        return "Iris-versicolor"
    
st.sidebar.title("Iris Flower Species Prediction App")

sepal_l = st.sidebar.slider("Sepal Length", float(iris_df["SepalLengthCm"].min()), float(iris_df["SepalLengthCm"].max()))
sepal_w = st.sidebar.slider("Sepal Width", float(iris_df["SepalWidthCm"].min()), float(iris_df["SepalWidthCm"].max()))
petal_l = st.sidebar.slider("Petal Length", float(iris_df["PetalLengthCm"].min()), float(iris_df["PetalLengthCm"].max()))
petal_w = st.sidebar.slider("Petal Width", float(iris_df["PetalWidthCm"].min()), float(iris_df["PetalWidthCm"].max()))

clf = st.sidebar.selectbox("Machine Learning Algorithm", ("Support Vector Machine", "Logistic Regression", "Random Forest Classifier"))

if st.sidebar.button("Predict"):
    if clf == "Support Vector Machine":
        model_pred = prediction(svc_model, sepal_l, sepal_w, petal_l, petal_w)
        score = svc_model.score(X_train, y_train)
    
    elif clf == "Logistic Regression":
        model_pred = prediction(log_reg, sepal_l, sepal_w, petal_l, petal_w)
        score = log_reg.score(X_train, y_train)
    
    else:
        model_pred = prediction(rf_clf, sepal_l, sepal_w, petal_l, petal_w)
        score = rf_clf.score(X_train, y_train)

    st.write("Species Predicted: ", model_pred)
    st.write("Accuracy Score of ", clf, " is ", score)