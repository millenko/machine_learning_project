import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(test_size=0.20, random_state=0):
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
    file_path = "../data/02_cleaned_data.csv"
    target_column = "is_canceled"
    
    df = pd.read_csv(file_path)
    
    features = df.drop(columns=[target_column])
    target = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)

    display(X_train.head())
    print("\n" * 2)  # Adds space between the outputs
    display(y_train.head())

    return X_train, X_test, y_train, y_test