import pandas as pd
import ast
from time import time
import importlib

# Define a dictionary mapping model names to their respective sklearn modules
model_module_dict = {
    'KNeighborsClassifier': 'sklearn.neighbors',
    'LogisticRegression': 'sklearn.linear_model',
    'DecisionTreeClassifier': 'sklearn.tree',
    'RandomForestClassifier': 'sklearn.ensemble',
    'GradientBoostingClassifier': 'sklearn.ensemble',
    'BaggingClassifier': 'sklearn.ensemble',
    'AdaBoostClassifier': 'sklearn.ensemble'
}

# Define a decorator to measure runtime
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        return result, int(end_time - start_time)
    return wrapper

@timer
def perform_training(model_class, params, scaled, X_train, X_test, y_train, X_train_scaled, X_test_scaled, y_test):
    model = model_class(**params)
    
    X_train_use = X_train_scaled if scaled else X_train
    X_test_use = X_test_scaled if scaled else X_test

    model.fit(X_train_use, y_train)
    accuracy = round(model.score(X_test_use, y_test) * 100, 2)
    pred = model.predict(X_test_use)
    
    return (params, accuracy, pred)

# Function to save summary results to CSV
def save_results_to_csv(model_name, best_params, accuracy, pred, runtime, file_name, scaled, y_test):
    results_summary = {
        "model": [model_name],
        "best_parameters": [best_params],
        "accuracy_in_%": [accuracy],
        "runtime_in_seconds": [runtime],
        "source": ["scaled" if scaled else "unscaled"]
    }
    summary_df = pd.DataFrame(results_summary)

    # Append results to CSV
    with open(file_name, 'a') as f:
        summary_df.to_csv(f, header=f.tell() == 0, index=False)

    # Display the DataFrame
    summary_df

# Function to process hyperparameter tuning results
def process_hyperparameter_tuning_results(input_file, output_file, X_train, X_test, y_train, X_train_scaled, X_test_scaled, y_test):
    # Load the CSV file containing hyperparameter tuning results
    hyperparameter_tuning_results = pd.read_csv(input_file)
    
    for idx, row in hyperparameter_tuning_results.iterrows():
        for scaled in [False, True]:
            source = 'scaled' if scaled else 'unscaled'
            print(f"Processing row {idx + 1}/{len(hyperparameter_tuning_results)} with {source} data")
            
            model_name = row['model']
            params = ast.literal_eval(row['best_parameters'])
            
            # Dynamically import the model class from sklearn
            try:
                module_path = model_module_dict[model_name]
                module = importlib.import_module(module_path)
                model_class = getattr(module, model_name)
                (best_params, accuracy, pred), runtime = perform_training(
                    model_class, params, scaled, X_train, X_test, y_train, X_train_scaled, X_test_scaled, y_test)
                save_results_to_csv(model_name, best_params, accuracy, pred, runtime, output_file, scaled, y_test)
            except (ImportError, AttributeError, KeyError) as e:
                print(f"Error importing model {model_name}: {e}")
                continue

# Example usage (uncomment to use directly in the script)
# process_hyperparameter_tuning_results(input_file, output_file, X_train, X_test, y_train, X_train_scaled, X_test_scaled, y_test)