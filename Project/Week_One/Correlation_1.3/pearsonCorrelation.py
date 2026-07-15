import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_path = "/home/lomi/Desktop/wqu_courses/wqu_course/Project/mexico-real-estate-combined-clean.csv"
data = pd.read_csv(data_path)

corr_area_price = (
        data["area_m2"].corr(data["price_usd"])
)

r = corr_area_price
r_sq = r**2
perc_var = r_sq*100
if __name__ == "__main__":
    print(f"the correlation between area and price is: {corr_area_price}")
    print (f"{perc_var}% of the variance in price can be explained by area")


# Correlation Matrix and Heatmaps

#Computing correlation matrix

corr_matrix= (
    data.select_dtypes("number").corr()
)
if __name__ == "__main__":
    print(corr_matrix)

# Heatmaps
# heatmaps encodes the correlation numbers with colors that we can immediately see the patterm
if __name__ == "__main__":
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap ="coolwarm", ax= ax)
    ax.set_title("Correlation Matrix of Realstate Data")
    plt.show()

# Note from the above heatmap- The diagonal — always 1.0 (deep red), because every variable is perfectly correlated with itself.    

# Segmented Correlation
if __name__ == "__main__":
    print(
        data
        .groupby("state")
        [["area_m2", "price_usd"]]
        .corr()
        .iloc[:,1]
        .xs("area_m2", level = 1)  
        # we can replace 'area_m2' with 'price_usd'
        .sort_values(ascending = False)
        .head(10)
        .tail(5)
    )


# Featuring Engeneering

df = (
    data
    .assign(price_per_m2 = lambda x: x["price_usd"] / x["area_m2"])
)
if __name__ == "__main__":
    print(df.head())