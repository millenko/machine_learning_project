#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import ast
from functools import wraps
import time
import pickle

def load_and_split_data(file_path="../data/02.csv", target_column="is_canceled", test_size=0.20, random_state=0):
    """
    Load data from a predefined CSV file, split it into training and testing sets,
    and return the resulting datasets.
    
    Parameters:
    - test_size: float, proportion of the dataset to include in the test split
    - random_state: int, seed used by the random number generator
    
    Returns:
    - X_train: pandas DataFrame, training features
    - X_test: pandas DataFrame, testing features
    - y_train: pandas Series, training target
    - y_test: pandas Series, testing target
    """
    df = pd.read_csv(file_path)
    features = df.drop(columns=[target_column])
    target = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled_np = scaler.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled_np, columns=X_train.columns)
    X_test_scaled_np = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_np, columns=X_test.columns)
    return X_train_scaled_df, X_test_scaled_df

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

@timer
def model_accuracy(model, X_train, y_train, X_test, y_test):
    model_name = str(model).split("(")[0]
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test) * 100
    return model_name, accuracy

if __name__ == "__main__":
    # Step 1: Load and Split Data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Step 2: Scale Data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Step 3: Train and Evaluate Models
    models = [
        KNeighborsClassifier(),
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        BaggingClassifier(),
        BaggingClassifier(KNeighborsClassifier()),
        BaggingClassifier(LogisticRegression()),
        BaggingClassifier(RandomForestClassifier()),
        BaggingClassifier(GradientBoostingClassifier()),
        AdaBoostClassifier(),
        AdaBoostClassifier(LogisticRegression()),
        AdaBoostClassifier(RandomForestClassifier()),
        AdaBoostClassifier(GradientBoostingClassifier())
    ]

    model_names = []
    accuracies = []
    times = []

    for model in models:
        (model_name, accuracy), time_taken = model_accuracy(model, X_train, y_train, X_test, y_test)
        model_names.append(model_name)
        accuracies.append(accuracy)
        times.append(time_taken)

    accuracies_unscaled = pd.DataFrame({
        "Model": model_names,
        "Accuracy, unscaled": [round(accuracy, 2) for accuracy in accuracies],
        "Runtime (seconds), unscaled": [int(time_taken) for time_taken in times]
    })

    print(accuracies_unscaled)

