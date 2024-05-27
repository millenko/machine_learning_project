from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    
    # Fit the scaler on the training data and transform it
    X_train_scaled_np = scaler.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled_np, columns=X_train.columns)
    
    # Transform the test data using the fitted scaler
    X_test_scaled_np = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_np, columns=X_test.columns)
    
    # Display the head of the scaled DataFrames with a separator
    display(X_train_scaled_df.head())
    print("\n" * 2)  # Adds space between the outputs
    display(X_test_scaled_df.head())
    
    return X_train_scaled_df, X_test_scaled_df

# Example usage (assuming X_train and X_test are already defined):
# X_train_scaled, X_test_scaled = scale_data(X_train, X_test)