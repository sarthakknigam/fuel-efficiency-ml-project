# Fuel Efficiency Prediction using Machine Learning

## Project Overview

This project predicts the **fuel efficiency (MPG - Miles Per Gallon)** of a vehicle using Machine Learning.
A trained model analyzes vehicle attributes from a dataset and predicts the expected fuel efficiency.

The project also includes a **simple HTML webpage interface** where users can input vehicle data and get predictions.

## Features

* Machine Learning model for fuel efficiency prediction
* Training script to train the model on dataset
* HTML webpage interface for prediction
* Dataset used for model training

## Project Structure

```
fuel-efficiency-ml-project
│
├── fuel_efficiency_dataset.csv   # Dataset used for training
├── train.py                      # Script to train ML model
├── improved_mpg_prediction.py    # Prediction model
└── fuel_prediction.html          # Web interface
```

## Technologies Used

* Python
* Machine Learning
* HTML
* NumPy / Pandas
* Scikit-learn

## How to Run the Project

### 1 Install dependencies

```
pip install pandas numpy scikit-learn
```

### 2 Train the model

```
python train.py
```

### 3 Run prediction script

```
python improved_mpg_prediction.py
```

### 4 Open Web Interface

Open the file:

```
fuel_prediction.html
```

in your browser.

## Dataset

The dataset contains vehicle information such as:

* Cylinders
* Horsepower
* Weight
* Acceleration
* Model Year

These features are used by the model to predict **fuel efficiency (MPG)**.

## Author

Sarthak Nigam
