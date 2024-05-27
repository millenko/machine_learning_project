import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Step 1: Load and Split Data
# For this example, we'll use the Iris dataset. Replace this with your actual data loading code.
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Save Scalers and Models

# Fit the scaler on the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Save the scaler to a file
with open("standard_scaler.pickle", "wb") as file:
    pickle.dump(scaler, file)

# Transform the training data
X_train_scaled_np = scaler.transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled_np, columns=X_train.columns)

# Train the KNN model on the scaled training data
model = KNeighborsClassifier()
model.fit(X_train_scaled_df, y_train)

# Save the trained KNN model to a file
with open("knn_5.pickle", "wb") as file:
    pickle.dump(model, file)

# Step 3: Load Scalers and Models for Testing and Transform Test Data

# Load the scaler from the file
with open("standard_scaler.pickle", "rb") as file:
    scaler = pickle.load(file)

# Load the trained KNN model from the file
with open("knn_5.pickle", "rb") as file:
    model = pickle.load(file)

# Transform the test data using the loaded scaler
X_test_scaled_np = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled_np, columns=X_test.columns)

# Evaluate the KNN model on the scaled test data
accuracy = model.score(X_test_scaled_df, y_test) * 100
print(f"The accuracy of the model is {accuracy:.2f}%")

# Step 4: Read User Data and Make Predictions

# For this example, we'll use a subset of the test data as new data
# In a real scenario, replace X_new with actual new data
X_new = X_test.sample(5, random_state=42)

# Transform the new data using the loaded scaler
X_new_scaled_np = scaler.transform(X_new)
X_new_scaled_df = pd.DataFrame(X_new_scaled_np, columns=X_new.columns)

# Make predictions on the scaled new data using the loaded KNN model
predictions = model.predict(X_new_scaled_df)
print(predictions)