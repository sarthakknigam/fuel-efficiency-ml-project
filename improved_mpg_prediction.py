# -----------------------------
# MOST ACCURATE FUEL PREDICTOR
# Using Optimized Ensemble Model
# -----------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
import joblib

# -----------------------------
# Load Dataset (for reference)
# -----------------------------
data = pd.read_csv("fuel_efficiency_dataset.csv")
data.dropna(inplace=True)

# -----------------------------
# Feature Engineering
# -----------------------------
data['power_to_weight'] = data['horsepower'] / data['weight']
data['displacement_per_cylinder'] = data['displacement'] / data['cylinders']
data['weight_per_cylinder'] = data['weight'] / data['cylinders']
data['power_displacement_ratio'] = data['horsepower'] / data['displacement']
data['weight_acceleration'] = data['weight'] * data['acceleration']
data['horsepower_squared'] = data['horsepower'] ** 2
data['weight_squared'] = data['weight'] ** 2
data['age'] = 2025 - data['model_year']

# Features & target
X = data[['cylinders', 'displacement', 'horsepower', 'weight', 
          'acceleration', 'model_year', 'power_to_weight', 
          'displacement_per_cylinder', 'weight_per_cylinder',
          'power_displacement_ratio', 'weight_acceleration',
          'horsepower_squared', 'weight_squared', 'age']]
y = data['mpg']

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train Ensemble (Optimized)
# -----------------------------
# Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.9,
    max_features='sqrt',
    random_state=42
)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)

# Fit individual models
gb_model.fit(X_scaled, y)
rf_model.fit(X_scaled, y)
ridge_model.fit(X_scaled, y)

# Ensemble
ensemble = VotingRegressor([
    ('gradient_boosting', gb_model),
    ('random_forest', rf_model),
    ('ridge', ridge_model)
])
ensemble.fit(X_scaled, y)

print("Ensemble trained successfully!")

# -----------------------------
# Function: Predict MPG
# -----------------------------
def predict_mpg(cylinders, displacement, horsepower, weight, acceleration, model_year):
    # Prepare input
    df = pd.DataFrame({
        'cylinders': [cylinders],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'weight': [weight],
        'acceleration': [acceleration],
        'model_year': [model_year]
    })

    # Derived features
    df['power_to_weight'] = df['horsepower'] / df['weight']
    df['displacement_per_cylinder'] = df['displacement'] / df['cylinders']
    df['weight_per_cylinder'] = df['weight'] / df['cylinders']
    df['power_displacement_ratio'] = df['horsepower'] / df['displacement']
    df['weight_acceleration'] = df['weight'] * df['acceleration']
    df['horsepower_squared'] = df['horsepower'] ** 2
    df['weight_squared'] = df['weight'] ** 2
    df['age'] = 2025 - df['model_year']

    # Scale features
    df_scaled = scaler.transform(df)

    # Predict using ensemble
    mpg_pred = ensemble.predict(df_scaled)
    return mpg_pred[0]

# -----------------------------
# EXAMPLE USAGE
# -----------------------------
mpg = predict_mpg(cylinders=4, displacement=200, horsepower=120, weight=2600, acceleration=15, model_year=2022)
print(f"Predicted MPG: {mpg:.2f}")