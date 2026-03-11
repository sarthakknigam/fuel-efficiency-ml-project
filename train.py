# ------------------------------------------
# Fuel Efficiency Prediction (Regression)
# ------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load the Auto MPG dataset (downloaded from UCI)
# --------------------------------------------------
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
data = pd.read_csv("fuel_efficiency_dataset.csv")


print(data.head())

# Remove missing values
data.dropna(inplace=True)

# --------------------------------------------------
# Select features and target
# --------------------------------------------------
# Features affecting fuel efficiency
X = data[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']]
y = data['mpg']

# --------------------------------------------------
# Split dataset (Training + Testing)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# --------------------------------------------------
# Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Train the Model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# Predict & Evaluate
# --------------------------------------------------
predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))
# Show first 10 predicted vs actual values
output = pd.DataFrame({
    'Actual MPG': y_test,
    'Predicted MPG': predictions
})

print(output.head(10))
# Example new vehicle data
new_data = pd.DataFrame({
    'cylinders': [4],
    'displacement': [200],
    'horsepower': [120],
    'weight': [2600],
    'acceleration': [15],
    'model_year': [2022]
})

# Scale values using the trained scaler
new_data_scaled = scaler.transform(new_data)

# Predict
new_prediction = model.predict(new_data_scaled)

print("Predicted MPG:", new_prediction[0])

# --------------------------------------------------
# Plot Results
# --------------------------------------------------
# --------------------------------------------------
# Plot Results
# --------------------------------------------------
plt.figure(figsize=(8,5))

# Actual MPG points
plt.scatter(range(len(y_test)), y_test, color='green', label="Actual MPG")

# Predicted MPG points
plt.scatter(range(len(y_test)), predictions, color='red', label="Predicted MPG")

plt.xlabel("Sample Index")
plt.ylabel("MPG")
plt.title("Actual vs Predicted Fuel Efficiency")
plt.legend()
plt.show()

