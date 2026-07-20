import pandas as pd
import numpy as np
import glob 
import os
import matplotlib.pyplot as plt
import missingno as msno


script_dir = os.path.dirname(os.path.abspath(__file__))
dataPath = os.path.join(script_dir, "buenos-aires-real-estate-*.csv")

# dataPath ="./wqu_course/Project/Week_Two/Data/buenos-aires-real-estate-*.csv"
# print(datasets)


datasets = glob.glob(dataPath)
file1= []
for x in datasets:
    data = pd.read_csv(x)
    file1.append(data)
"""for the above one we can also use the list comprhension method"""
file2 = [pd.read_csv(x)for x in datasets] 
df = pd.concat(file1, ignore_index= True)
# print(df.head())


def merged_files(data):
    return pd.concat([pd.read_csv(file) for file in glob.glob(data)], ignore_index =True)

# print(merged_files(dataPath))
def filter_df(data):
    return(
        df.loc[ lambda x: x["place_with_parent_names"].str.contains("Capital Federal")]
        .query('property_type == "apartment"')
        .query('price_aprox_usd <400_000')
    )

"""Handling Outliners"""
def outliners_df(data):
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



df_clean = (
    df
    .pipe(filter_df)
    .pipe(outliners_df)
    .pipe(modify_cols)
    .pipe(modify_cols)
)
if __name__ == "__main__":
    """Missing Values Visualization"""
    # Percentage of missing values by column
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