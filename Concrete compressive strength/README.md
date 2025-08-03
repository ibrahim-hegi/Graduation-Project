# Concrete Compressive Strength Prediction

This project aims to predict the compressive strength of concrete using various machine learning models, with a focus on the XGBoost Regressor optimized through Grid Search. The dataset includes features such as cement, water, coarse aggregate, fine aggregate, and age, among others, to predict the target variable, concrete compressive strength.

## Project Overview

The notebook `Concrete Compressive Strength.ipynb` explores multiple regression models, evaluates their performance, and identifies the best model for predicting concrete compressive strength. The key steps include:

1. **Data Preprocessing**: Loading and preparing the dataset for modeling.
2. **Model Training**: Training several models, including Linear Regression, Lasso, Ridge, SVR, Random Forest, and XGBoost.
3. **Hyperparameter Tuning**: Using Grid Search to optimize the XGBoost Regressor.
4. **Model Evaluation**: Comparing models based on Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score, and Cross-Validated RMSE.
5. **Prediction**: Making predictions on new data using the best model.
6. **Model Saving**: Saving the optimized XGBoost model for future use.

## Dependencies

To run the notebook, install the required Python packages:
pip install numpy pandas matplotlib scikit-learn xgboost

## Load the Saved Model
import pickle
model = pickle.load(open('model.pkl', 'rb'))

## Model Performance

The models were evaluated based on Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score, and Cross-Validated RMSE. Below is a summary of the results:

| Model                     | MAE       | MSE        | R² Score | RMSE (Cross Validated) |
|---------------------------|-----------|------------|----------|------------------------|
| XGBRegressor (GridSearch) | 2.8279    | 19.7840    | 0.9337   | 9.1701                |
| XGBRegressor              | 2.7382    | 20.5965    | 0.9310   | 9.6405                |
| RandomForestRegressor     | 3.4732    | 27.3795    | 0.9082   | 10.2781               |
| Ridge                     | 8.6850    | 121.2165   | 0.5937   | 10.7492               |
| LinearRegression          | 8.6831    | 121.1918   | 0.5938   | 10.7538               |
| Lasso                     | 9.0980    | 133.5093   | 0.5525   | 10.9195               |
| SVR                       | 8.2723    | 115.9420   | 0.6114   | 11.5979               |

The **XGBRegressor (GridSearch)** achieved the best performance with the lowest RMSE and highest R² Score.




