import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from Feature_Engineering import clean_files

from category_encoders import OneHotEncoder
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir,"Data", "buenos-aires-real-estate-*.csv")
clean_df = clean_files(file_path)
target_col = "price_aprox_usd"
features_col =[col for col in clean_df.columns if col not in target_col]
X = clean_df[features_col]
y = clean_df[target_col]
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {X.columns.tolist()}")