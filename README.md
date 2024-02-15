# Housing_Price_Prediction
Comparison of Housing Price Prediction using SVM, RA and LRM Model Techniques
Here is a more detailed README for the housing price prediction document:

## Introduction

Predicting housing prices is an important real-world regression problem that can guide investment decisions. This project aimed to compare different machine learning techniques for predicting housing sale prices.

The models were trained and evaluated on the Boston Housing dataset, which contains attributes of houses in the Boston area like number of rooms, location, age, highway access, crime rate, etc. The target variable to predict was the median housing price.

## Data Preprocessing

The Python Pandas library was used to load and analyze the dataset. Data visualization included histograms, scatter plots, and line plots to understand relationships between variables. 

Data preprocessing steps:

- Handling missing values
- Converting data types
- Feature engineering 
- Normalization techniques like standard scaling

## Modeling

The data was split into training and test sets for model fitting and evaluation. The following regression models were compared:

- **Linear Regression** - Baseline model, assumes linear relationship between predictors and target. Grid search used for regularization hyperparameter tuning.

- **Random Forest** - Ensemble method, trains many decision trees on bootstrapped data samples. Key hyperparameters tuned were number of trees and maximum depth. Random subsets of features considered for splits to decorrelate trees.

- **Support Vector Machine (SVM)** - Flexible non-linear model, kernel methods used to project data into higher dimensions. Both linear and Radial Basis Function (RBF) kernels evaluated. Regularization strength and insensitive loss tuned.  

## Evaluation

Model performance was evaluated on the test set using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared

These metrics were used to compare models and select the best approach. Feature importance was also analyzed for random forest.

## Results

The random forest model achieved the lowest RMSE and highest accuracy of all models. Tuning the hyperparameters maximized performance. The nonlinear RBF kernel SVM also performed well. The linear regression model was significantly worse.

## Conclusion

The ensemble approach of random forests excelled for this regression task. Bagging, feature randomness, and averaging across many decorrelated decision trees prevented overfitting and captured complex relationships. The results highlight the importance of model selection and hyperparameter tuning for machine learning.

Overall, this project provided a useful workflow and comparison of regression techniques for an important prediction problem.

