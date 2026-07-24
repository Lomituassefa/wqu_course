import pandas as pd
import numpy as np
import glob 
import os
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Data", "buenos-aires-real-estate-*.csv")
def merged_files(file_path):
    return pd.concat([pd.read_csv(file) for file in glob.glob(file_path)], ignore_index =True)
# print(merged_files(dataPath))
def filter_df(data):
    return(
        data.loc[ lambda x: x["place_with_parent_names"].str.contains("Capital Federal")]
        .query('property_type == "apartment"')
        .query('price_aprox_usd <400_000')
    )
"""Handling Outliners"""
def outliers_df(data):
    return (
        data.loc[lambda x : x["surface_covered_in_m2"].between(
            x["surface_covered_in_m2"].quantile(0.1),
            x["surface_covered_in_m2"].quantile(0.9)
        )]
    )

"""Extract features from strings"""
def modify_cols(data):
    return (
        data
        .assign(
            lat=lambda x: x["lat-lon"].str.split(",", expand=True)[0].astype(float),      # <--- "lat-lon"
            lon=lambda x: x["lat-lon"].str.split(",", expand=True)[1].astype(float),      # <--- "lat-lon"
            neighborhood=lambda x: x["place_with_parent_names"].str.split("|", expand=True)[3]            # <--- "place_with_parent_names"
        )
    )


"""Chain all the dataframes on the cleaning method of the above functions"""
def clean_files(file_path):
    """Final function to complete the cleaning process"""
    drop_cols = [
        "lat-lon", "place_with_parent_names",
        "floor", "expenses", "rooms", "price",
        "price_aprox_local_currency", "price_usd_per_m2", 
        "price_per_m2", "operation", "property_type", "currency", 
        "properati_url", "surface_total_in_m2"
    ]

    return (
        merged_files(file_path)
        .pipe(filter_df)
        .pipe(outliers_df)
        .pipe(modify_cols)
        .drop(columns=drop_cols)
        .dropna()

    )
if __name__ == "__main__":
    """Missing Values Visualization"""
    # Percentage of missing values by column
    df = merged_files(file_path)
    nan_columns_percentage = (df.isna().sum(axis=0) * 100 / len(df)).sort_values()
    fig, ax = plt.subplots()
    # Horizontal bar plot
    ax.barh(y=nan_columns_percentage.index, width=nan_columns_percentage)
    ax.set_title('Percentage of Missing Values by Column')
    ax.set_xlabel('Percentage (%)')
    ax.set_ylabel('Column')
    # plt.show()
    # Plot missing values with color royalblue
    fig, ax = plt.subplots()
    msno.matrix(df, color=(0.3, 0.4, 0.8), ax=ax)
    plt.show()

    corr_matrix = df.select_dtypes(include= "number").corr()
    fig, ax = plt.subplots(figsize=(14,12))
    sns.heatmap(corr_matrix, annot = True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    plt.show()