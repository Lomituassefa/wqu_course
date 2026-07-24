import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Feature_Engineering import clean_files
from sklearn.model_selection import train_test_split
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir,"Data", "buenos-aires-real-estate-*.csv")
clean_df = clean_files(file_path)
"Multi Collinearity "
corr_matrix = clean_df.select_dtypes(include= "number").corr()
fig, ax = plt.subplots(figsize=(14,12))
sns.heatmap(corr_matrix, annot = True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.show()
                 # <--- Drop rows with missing values


""" Target and  Features 'Linear Regression'"""

target = "price_aprox_usd"
features = [cols for cols in clean_df.columns if cols not in target]
y = clean_df[target]
X = clean_df[features]

# Encoding catergorical variables: One-Hot encoding 
sample_df = (
    clean_df[["neighborhood", "price_aprox_usd"] ]
)
#  Transform the categorical colomun 'neighborhood' binary column and drop the existing column of category to avoid data redunduncy 
encoded_df = (
    sample_df
    .assign(
        is_palermo = (sample_df["neighborhood"] == "Palermo").astype(int),  # <--- Create binary column: 1 if Palermo, 0 otherwise
        is_caballito = (sample_df["neighborhood"] =="Caballito" ).astype(int),  # <--- Create binary column: 1 if Caballito, 0 otherwise
        is_villa_luro = (sample_df["neighborhood"] =="Villa Luro" ).astype(int),  # <--- Create binary column: 1 if Villa Luro, 0 otherwise
    )
    .drop(columns=['neighborhood'])  # <--- Remove the original text column (we no longer need it)
)
"""Linear Regration introduction"""
# Split into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # <--- X, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

