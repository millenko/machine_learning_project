import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
import time

# Decorator to measure the time of function execution
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = int(time.perf_counter() - start_time)
        return result, runtime
    return wrapper

# Function to perform search
@timer
def perform_search(model, param_grid, search_type, X_train, y_train, X_test, y_test, X_train_scaled=None, X_test_scaled=None, scaled=False):
    """
    Perform hyperparameter search for a given model using GridSearchCV or RandomizedSearchCV.
    
    Parameters:
    - model: The machine learning model to be tuned.
    - param_grid: The hyperparameter grid for the model.
    - search_type: The type of search ('grid' for GridSearchCV, 'random' for RandomizedSearchCV).
    - X_train: Training features.
    - y_train: Training labels.
    - X_test: Test features.
    - y_test: Test labels.
    - X_train_scaled: Scaled training features (optional).
    - X_test_scaled: Scaled test features (optional).
    - scaled: Boolean indicating whether to use scaled data.
    
    Returns:
    - best_params: Best hyperparameters found.
    - accuracy: Accuracy of the best model on the test set.
    - pred: Predictions of the best model on the test set.
    """
    X_train_use = X_train_scaled if scaled else X_train
    X_test_use = X_test_scaled if scaled else X_test
    
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    elif search_type == 'random':
        search = RandomizedSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, n_iter=30)
    else:
        raise ValueError("search_type must be either 'grid' or 'random'")
    
    search.fit(X_train_use, y_train)
    accuracy = round(search.score(X_test_use, y_test) * 100, 2)
    pred = search.best_estimator_.predict(X_test_use)
    return search.best_params_, accuracy, pred

# Function to save summary results to CSV
def save_results_to_csv(model_name, best_params, accuracy, pred, runtime, file_name, y_test, scaled, search_type):
    """
    Save summary results of the hyperparameter search to a CSV file.
    
    Parameters:
    - model_name: Name of the model.
    - best_params: Best hyperparameters found.
    - accuracy: Accuracy of the best model on the test set.
    - pred: Predictions of the best model on the test set.
    - runtime: Time taken for the hyperparameter search.
    - file_name: Name of the CSV file to save results.
    - y_test: Test labels.
    - scaled: Boolean indicating whether scaled data was used.
    - search_type: The type of search ('grid' or 'random').
    """
    results_summary = {
        "model": [model_name],
        "best_parameters": [best_params],
        "accuracy_in_%": [accuracy],
        "runtime_in_seconds": [runtime],
        "source": [f"{search_type}_search_scaled" if scaled else f"{search_type}_search_unscaled"],
        "classification_report": [classification_report(y_true=y_test, y_pred=pred, output_dict=True)]
    }
    summary_df = pd.DataFrame(results_summary)

    # Append results to CSV
    with open(file_name, 'a') as f:
        summary_df.to_csv(f, header=f.tell() == 0, index=False)

# Helper function to perform search and save results
def search(model_class, model_name, param_grids, search_type, X_train, y_train, X_test, y_test, X_train_scaled=None, X_test_scaled=None, scaled=False, file_name=None):
    """
    Perform hyperparameter search and save results for a given model class and parameter grid.
    
    Parameters:
    - model_class: The machine learning model class to be tuned.
    - model_name: The name of the model.
    - param_grids: The hyperparameter grids for different models.
    - search_type: The type of search ('grid' or 'random').
    - X_train: Training features.
    - y_train: Training labels.
    - X_test: Test features.
    - y_test: Test labels.
    - X_train_scaled: Scaled training features (optional).
    - X_test_scaled: Scaled test features (optional).
    - scaled: Boolean indicating whether to use scaled data.
    - file_name: Name of the CSV file to save results (optional).
    """
    param_grid = param_grids.get(model_name)
    if param_grid is None:
        raise ValueError(f"No parameter grid found for {model_name}")

    model_kwargs = {"random_state": 42} if "random_state" in model_class().get_params() else {}
    model = model_class(**model_kwargs)
    
    (best_params, accuracy, pred), runtime = perform_search(model, param_grid, search_type, X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, scaled)
    
    if file_name is None:
        file_name = "../data/hyperparameter_tuning_results.csv"
    
    save_results_to_csv(model_name, best_params, accuracy, pred, runtime, file_name, y_test, scaled, search_type)
    
    print(f"Model: {model_name}")
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy of the tuned model: {accuracy}%")
    print(f"Runtime in seconds: {runtime}\n")
    print(classification_report(y_true=y_test, y_pred=pred))

"""
Example usage (assuming param_grids and models are defined elsewhere)

1. Single model evaluation:
search(KNeighborsClassifier, "KNeighborsClassifier", param_grids, search_type='grid', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scaled=False)
search(AdaBoostClassifier, "AdaBoostClassifier", param_grids, search_type='random', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scaled=True)

2. Loop through models and perform grid and/or randomized searches:
for model_class in models:
    model_name = model_class.__name__
    
    # Perform grid search with scaled and unscaled data
    search(model_class, model_name, param_grids, search_type='grid', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scaled=False)
    search(model_class, model_name, param_grids, search_type='grid', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scaled=True)
    
    # Perform randomized search with scaled and unscaled data
    search(model_class, model_name, param_grids, search_type='random', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scaled=False)
    search(model_class, model_name, param_grids, search_type='random', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scaled=True)
"""