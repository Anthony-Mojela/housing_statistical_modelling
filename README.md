statistical-modelling-housing-prices-r/

Project Overview
This project applies statistical regression modelling techniques to predict housing prices. The main objective is to build a rigorous modelling pipeline grounded in statistical principles, and to evaluate how regularised regression and modern predictive models compare to classical linear regression.

Problem Statement
House price prediction is a core regression problem in real-estate analytics. In this project, I model housing prices using ordinary least squares regression, assess its assumptions, and extend the framework to regularised regression and nonlinear models to improve predictive performance and generalisation.

Dataset
Housing dataset
Target variable: House price (log-transformed to improve normality)
Predictors: multiple structural and property-related features

Tools & Libraries
R, ggplot2, dplyr, recipes, caret, MLmetrics, H2O, lmtest, rpart

Statistical Modelling Workflow
1. Exploratory analysis & target transformation
Distributional assessment of the response variable
Log-transformation to better satisfy linear model assumptions
2. Data preprocessing
Train/test split (80/20)
Feature normalisation
Categorical encoding using recipes
3. Classical regression modelling
Ordinary Least Squares (OLS)
Model inference and coefficient interpretation
Residual diagnostics and Durbin-Watson autocorrelation test
4. Regularised regression
Ridge regression
Lasso regression
Elastic Net
Hyperparameter tuning using cross-validation
5. Predictive extensions (for comparison)
Regression trees
K-Nearest Neighbours
Support Vector Regression
Neural networks
6. Evaluation framework
MAE, RMSE, R²
Key Outcomes
Train vs test performance
Visual model comparison

Key Outcomes
Built a complete statistical modelling pipeline
Demonstrated how regularisation improves model stability
Compared classical regression with flexible predictive models
Developed a reusable evaluation framework across models
