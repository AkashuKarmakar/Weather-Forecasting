                                    -- Weather Prediction Project--

This project focuses on forecasting atmospheric temperature using machine learning models, specifically Random Forest and Long Short-Term Memory (LSTM) networks. It includes data preprocessing, extensive feature engineering, model training, evaluation, and multi-step prediction capabilities.

	*Table of Contents*

	Project Overview
	Features
	Usage
	Model Details
	Evaluation Metrics
	File Descriptions

	*Project Overview*

This repository contains a robust weather prediction system designed to forecast atmospheric temperature for up to three days in advance. The system leverages historical atmospheric variable data, performs comprehensive data preprocessing and feature engineering, and utilizes advanced machine learning models (Random Forest and LSTM) for accurate predictions. It also includes tools for model evaluation, interpretability, and drift analysis.

*Features*

Data Preprocessing: Loads and processes atmospheric variables (Geopotential Height (HGT), Relative Humidity (RH), Temperature (TMP), Zonal Wind (UGRD), Meridional Wind (VGRD)) from NetCDF files.
Feature Engineering: Generates lag features, rolling means, and time-based features (month, day of year) to enhance model performance.
Multi-Step Prediction: Capable of forecasting temperature for t+1, t+2, and t+3 days.

Model Training:

Random Forest Regressor: Includes hyperparameter tuning using RandomizedSearchCV.
LSTM Neural Network: Utilizes a deep learning approach for sequential data forecasting.
Model Evaluation: Assesses model performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) scores.
Baseline Model Comparison: Compares the advanced models against a simple baseline model (predicting tomorrow's temperature as today's).
Model Explainability: Employs SHAP (SHapley Additive exPlanations) values to interpret Random Forest model predictions.
Model Robustness Analysis: Includes cross-validation and model drift analysis to ensure reliability over time.
Prediction Script: A dedicated script (prediction.ipynb) for making new predictions using saved models.

	*Setup*

Clone the repository (if applicable):

Bash

git clone <repository_url>
cd <repository_name>
Data Acquisition:

Obtain the atmospheric variable data (NetCDF files for HGT, RH, TMP, UGRD, VGRD) and place them in the specified base_path (e.g., F:/projects/personal project/data/Atmospheric_variable/) as configured in weather_prediction1.0.ipynb. The data should be organized into folders like "IMDAA_HGT_prl_1.08_1990_2020", "IMDAA_RH_prl_1.08_1990_2020", etc..
Install Dependencies:

The project requires the following Python libraries:

pandas
numpy
xarray
tqdm
scikit-learn
tensorflow
keras (if using an older version of TensorFlow, otherwise tensorflow.keras is sufficient)
joblib
matplotlib
seaborn
shap
You can install them using pip:

Bash

pip install pandas numpy xarray tqdm scikit-learn tensorflow joblib matplotlib seaborn shap

	*Usage*

1. Data Preprocessing and Initial Model Training (weather_prediction1.0.ipynb)
Run this notebook first to preprocess the raw atmospheric data and perform an initial training and evaluation of Random Forest and LSTM models for single-step prediction. This will generate weather_atmospheric_processed.csv.

2. Advanced Model Training and Analysis (weather_prediction2.0.ipynb)
Execute this notebook to:

Perform comprehensive feature engineering.
Train Random Forest and LSTM models for multi-step (t+1, t+2, t+3) temperature predictions.
Conduct hyperparameter tuning, evaluation, baseline comparison, SHAP analysis, cross-validation, and model drift analysis.
Save the trained models and the MinMaxScaler. The models and scaler will be saved in the models/ directory (e.g., F:/projects/personal project/weather prediction_v1/models/).
3. Making New Predictions (prediction.ipynb)
After training the models using weather_prediction2.0.ipynb, use prediction.ipynb to make new temperature predictions for the next three days. Ensure that the weather_atmospheric_processed.csv and the saved models/scaler are in their expected paths.

	*Model Details*

Random Forest Regressor: An ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the mean prediction of the individual trees. Hyperparameters are tuned using RandomizedSearchCV.
Long Short-Term Memory (LSTM): A type of recurrent neural network (RNN) well-suited for sequence prediction problems due to its ability to learn long-term dependencies. 
The models are uploaded at 'https://huggingface.co/AkashuKarmakar/Random_forest_weather_forecast/tree/main'

	*Evaluation Metrics*

The models are evaluated using the following metrics:

Mean Absolute Error (MAE): Measures the average magnitude of the errors in a set of predictions, without considering their direction.
Root Mean Squared Error (RMSE): Measures the square root of the average of the squared errors, giving higher weight to larger errors.
R-squared (R² Score): Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

	*File Descriptions*

prediction.ipynb: This notebook is used for making real-time predictions for the next 3 days using the trained Random Forest and LSTM models.
weather_prediction1.0.ipynb: Handles initial data preprocessing, loading atmospheric variables, and basic single-step model training and evaluation for Random Forest and LSTM.
weather_prediction2.0.ipynb: Contains advanced feature engineering, multi-step prediction model training (Random Forest with hyperparameter tuning, LSTM), and in-depth analysis including evaluation, baseline comparison, SHAP, cross-validation, and model drift.
