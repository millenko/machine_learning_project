# Standard Library Imports
import ast
from functools import wraps
import time

# Third-Party Imports
import pandas as pd
# Set display options for pandas DataFrames
# This ensures the full width of columns is displayed, which is useful for viewing long string values (Machine Learning: best_parameters)
pd.set_option('display.max_colwidth', None)

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

# Local Application/Library Specific Imports
from utils import timer