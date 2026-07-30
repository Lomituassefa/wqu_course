import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from Feature_Engineering import clean_files
from category_encoders import OneHotEncoder
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) 
clean_df = clean_files()
target_col = "price_aprox_usd"
features_col =[col for col in clean_df.columns if col not in target_col]
X = clean_df[features_col]
y = clean_df[target_col]
"""Ridge Rigression Pipeline"""
# Without pipeline 
ohe = OneHotEncoder(use_cat_names=True)
scaler = StandardScaler()
X_encoded = ohe.fit_transform(X)
X_scaled = scaler.fit_transform(X_encoded)

# X_train, X_test, y_train, y_test = train_test_split(X_scaled,y)
#With pipeline
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
model_ridge = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    StandardScaler(),
    Ridge(alpha=1.0)
)
model_ridge.fit(X_train, y_train)
y_pred_ridge_train = model_ridge.predict(X_train)
rmse_ridge_train = root_mean_squared_error(y_train, y_pred_ridge_train)

y_pred_ridge_test = model_ridge.predict(X_test)
rmse_ridge_test = root_mean_squared_error(y_test, y_pred_ridge_test)


model_lesso = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    StandardScaler(),
    Lasso(alpha=100, max_iter=10000)
)
model_lesso.fit(X_train, y_train)
y_pred_lesso_train = model_lesso.predict(X_train)
rmse_lesso_train = root_mean_squared_error(y_train, y_pred_lesso_train)

y_pred_lesso_test = model_lesso.predict(X_test)
rmse_lesso_test= root_mean_squared_error(y_test, y_pred_lesso_test)
print(rmse_lesso_test, rmse_lesso_train)

"""Bias-variance tradeoff"""
# For ridge 
alphas = [0.001, 0.01, 0.1,1, 10, 100, 1000]
ridge_results = []
for a in alphas:
    model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    StandardScaler(),
    Ridge(alpha=a)
    )
    model.fit(X_train, y_train)
    rmse_train = root_mean_squared_error(y_train, model.predict(X_train))
    rmse_test = root_mean_squared_error(y_test, model.predict(X_test))
    ridge_results.append({
        "alpha":a,
        "rmse_train" : rmse_train,
        "rmse_test" : rmse_test,
    })

df_ridge_results = pd.DataFrame(ridge_results)
"""Ridge bias-variance tradeoff plot 'U shaped'. training and rest RMSE
are ploted on the y axis when the alpha is on the x axis"""

fig1, ax = plt.subplots(figsize=(9,6))
ax.plot(df_ridge_results["alpha"], df_ridge_results["rmse_train"],marker='o', linewidth=2, markersize=8, label="Training RMSE (Low Bias)")
ax.plot(df_ridge_results["alpha"], df_ridge_results["rmse_test"],marker='s', linewidth=2, markersize=8, label="Test RMSE (Generalization)")
ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, label="α=1 (Our choice)")
ax.set_ylabel("RMSE ($)", fontsize=12)
ax.set_xscale("log")
ax.set_title("Ridge Regression: The Bias-Variance Tradeoff", fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(os.path.join(script_dir, "Ridge_TradeOff.png"), dpi=100)
plt.close(fig1) 

# For Lasso
lasso_results = []
alphas_lasso = [0.1, 1, 10, 100, 1000]

for a in alphas_lasso:
    model = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        StandardScaler(),
        Lasso(alpha=a, max_iter=10000)
    )
    model.fit(X_train, y_train)  

    rmse_train = root_mean_squared_error(y_train, model.predict(X_train)) 
    rmse_test = root_mean_squared_error(y_test, model.predict(X_test)) 
    non_zero_features = (model.named_steps["lasso"].coef_ != 0).sum()

    lasso_results.append({
        "alpha": a,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "non_zero_features": non_zero_features
    })

df_lasso_results = pd.DataFrame(lasso_results)
# Ridge vs Lasso
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
ax1.plot(df_ridge_results["alpha"], df_ridge_results["rmse_test"],
         marker='o', linewidth=2, markersize=8, label="Ridge Test RMSE")
ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label="α=1 (Ridge default)")
ax1.set_xlabel("alpha (Regularization Strength)")
ax1.set_ylabel("Test RMSE ($)")
ax1.set_xscale("log")
ax1.set_title("Ridge: Alpha vs Test RMSE")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Lasso comparison with feature count
ax2_twin = ax2.twinx()
ax2.plot(df_lasso_results["alpha"], df_lasso_results["rmse_test"],
         marker='s', linewidth=2, markersize=8, color='green', label="Lasso Test RMSE")
ax2_twin.plot(df_lasso_results["alpha"], df_lasso_results["non_zero_features"],
              marker='^', linewidth=2, markersize=8, color='orange', linestyle='--', label="Non-zero Features")
ax2.axvline(x=100, color='red', linestyle='--', alpha=0.7, linewidth=2, label="α=100 (Our choice)")
ax2.set_xlabel("alpha (Regularization Strength)")
ax2.set_ylabel("Test RMSE ($)", color='green')
ax2_twin.set_ylabel("Number of Non-zero Features", color='orange')
ax2.set_xscale("log")
ax2.set_title("Lasso: Alpha vs Test RMSE & Feature Count")
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(os.path.join(script_dir, "Ridge_vS_Lasso.png"), dpi=100)
plt.close(fig2) 

"""Cross validation and automated alpha selection with ridgeCV  and LassoCV"""
# for ridge 
"""RidgeCV(alphas=[...]) automatically selects the best 
 from the provided list using cross-validation. 
 The chosen alpha is stored in model.alpha_ after fitting."""
ridge_cv_model = make_pipeline(
    OneHotEncoder(use_cat_names= True),
    StandardScaler(),
    RidgeCV(alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000], cv = 5)
)
ridge_cv_model.fit(X_train, y_train)
optimal_alpha_ridge = ridge_cv_model.named_steps["ridgecv"].alpha_
# for lasso
lasso_cv_model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    StandardScaler(),
    LassoCV(alphas=[0.1, 1, 10, 100, 1000], cv=5, max_iter=10000)
)
lasso_cv_model.fit(X_train, y_train)
optimal_alpha_lasso = lasso_cv_model.named_steps["lassocv"].alpha_