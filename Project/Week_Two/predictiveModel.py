import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from Feature_Engineering import clean_files
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir,"Data", "buenos-aires-real-estate-*.csv")

df = clean_files(file_path)
# print(df.info())
"""2.2.1.1 Creating the Feature Matrix and Target Vector"""
# the double bracket for the feature matrix 'X' is to create a 2D matrix 
X = df[["surface_covered_in_m2"]]
y = df["price_aprox_usd"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
# The baseline: it the the naive prediction we could make if we have no modeling. which would be the mean price across all training examples
y_mean =  y_train.mean()
y_pred_baseline = [y_mean]*len(y_test)
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)


model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
# Training performance
y_pred_train = model_lr.predict(X_train)
mae_train = mean_absolute_error(y_train,y_pred_train)
rmse_train = root_mean_squared_error(y_train,y_pred_train)
r2_train = r2_score(y_train,y_pred_train)

# Test Perforance
y_pred_test = model_lr.predict(X_test)  # <--- model_lr.predict(X_test)
mae_test = mean_absolute_error(y_test,y_pred_test)  # <--- mean_absolute_error(y_test, y_pred_test)
rmse_test = root_mean_squared_error(y_test,y_pred_test)  # <--- root_mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test,y_pred_test)
# Comparison between training and test one
metrics_comparison = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R²'],
    'Training': [ mae_train, rmse_train, r2_train],
    'Test': [mae_test, rmse_test, r2_test] 
})
# print(metrics_comparison)
intercept = model_lr.intercept_  
coefficient = model_lr.coef_[0] 

fig1, axes = plt.subplots(1,2,figsize = (14,5))
# Trainig set
axes[0].scatter(y_train, y_pred_train, alpha=0.5)
axes[0].plot([y_train.min(), y_train.max()],
             [y_train.min(), y_train.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price (USD)')
axes[0].set_ylabel('Predicted Price (USD)')
axes[0].set_title('Training Set: Predicted vs Actual')
axes[0].legend()
# Test set
axes[1].scatter(y_test, y_pred_test, alpha=0.5, color='green')
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Price (USD)')
axes[1].set_ylabel('Predicted Price (USD)')
axes[1].set_title('Test Set: Predicted vs Actual')
axes[1].legend()
plt.tight_layout()
fig1.savefig(os.path.join(script_dir, "predicted_vs_actual.png"), dpi=100)
plt.close(fig1) 

"""Residual Analysis"""
residuals_train = y_train - y_pred_train 
residuals_test = y_test - y_pred_test 
fig2, axes = plt.subplots(1, 2, figsize=(9, 6))
# Training residuals
axes[0].scatter(y_pred_train, residuals_train, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Price (USD)')
axes[0].set_ylabel('Residuals (USD)')
axes[0].set_title('Training Set: Residual Plot')
# Test residuals
axes[1].scatter(y_pred_test, residuals_test, alpha=0.5, color='green')
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Price (USD)')
axes[1].set_ylabel('Residuals (USD)')
axes[1].set_title('Test Set: Residual Plot')
plt.tight_layout()
fig2.savefig(os.path.join(script_dir, "residuals.png"), dpi=100)
plt.close(fig2)
"""Line of Bestfit"""

fig3, ax = plt.subplots()
ax.scatter(X_train, y_train, alpha=0.5, label='Training Data')
ax.scatter(X_test, y_test, alpha=0.5, color='green', label='Test Data')
# Create evenly-spaced x values for the line
x_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
ax.plot(x_line, model_lr.predict(x_line), color="red", linewidth=2, label="Linear Model")
ax.set_xlabel("Surface Area [sq meters]")
ax.set_ylabel("Price [USD]")
ax.set_title("Buenos Aires: Price vs. Surface Area")
ax.legend()
plt.tight_layout()
fig3.savefig(os.path.join(script_dir, "bestfit.png"), dpi=100)
plt.close(fig3)