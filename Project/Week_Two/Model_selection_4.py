import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from category_encoders import OneHotEncoder
from Feature_Engineering import clean_files
df = clean_files()
target = "price_aprox_usd"
feature = [col for col in df.columns if col != target]
X = df[feature]
y = df[target]
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X,y, test_size=0.2, random_state = 42)
def get_rmse_4(model,X,y_true):
    y_pred = model.predict(X)
    rmse = root_mean_squared_error(y_true, y_pred)
    return rmse
# Baseline Model 
y_mean_4 = y_train_4.mean()
y_pred_baseline_train = [y_mean_4]*len(y_train_4)
y_pred_baseline_test = [y_mean_4]*len(y_test_4)
rmse_base_train = root_mean_squared_error(y_train_4, y_pred_baseline_train) 
rmse_base_test = root_mean_squared_error(y_test_4,y_pred_baseline_test) 
# print(f"Baseline Train RMSE: ${rmse_base_train:,.2f}")
# print(f"Baseline Test RMSE: ${rmse_base_test:,.2f}")
# Linear Regression
model_linear_4 = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    StandardScaler(),
    LinearRegression()
)
model_linear_4.fit(X_train_4,y_train_4)
rmse_linear_train = get_rmse_4(model_linear_4,X_train_4,y_train_4) 
rmse_linear_test = get_rmse_4(model_linear_4,X_test_4, y_test_4)
# Ridge Regression(CV)

model_ridge_4 = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-2,4,100)) 
)
model_ridge_4.fit(X_train_4,y_train_4) 
rmse_ridge_train = get_rmse_4(model_ridge_4,X_train_4,y_train_4)
rmse_ridge_test = get_rmse_4(model_ridge_4,X_test_4,y_test_4)
alpha_ridge_chosen = model_ridge_4.named_steps["ridgecv"].alpha_
# Lasso Regression
model_lasso_4 = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    StandardScaler(),
    LassoCV(alphas=np.logspace(-4, 2, 100), max_iter=10000)
)

model_lasso_4.fit(X_train_4,y_train_4)
rmse_lasso_train = get_rmse_4(model_lasso_4,X_train_4,y_train_4)
rmse_lasso_test = get_rmse_4(model_lasso_4, X_test_4,y_test_4) 
alpha_lasso_chosen = model_lasso_4.named_steps["lassocv"].alpha_
print(f"Best Lasso alpha: {alpha_lasso_chosen:.6f}")
print(f"Lasso Train RMSE: ${rmse_lasso_train:,.2f}")
print(f"Lasso Test RMSE: ${rmse_lasso_test:,.2f}")
