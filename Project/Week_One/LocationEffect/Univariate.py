import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df =(
    pd.read_csv("/home/lomi/Desktop/wqu_courses/wqu_course/Project/mexico-real-estate-combined-clean.csv")
    .assign(price_per_m2 = lambda x: x["price_usd"]/x["area_m2"])
)

fig, ax = plt.subplots()

ax.hist(df["price_per_m2"], bins = 30, edgecolor = "white")

ax.set_title("Distribution of Price per Square meter")
ax.set_xlabel("Price [USD]")
ax.set_ylabel("Frequency")

# Adding a vertical line for the median

median_price_per_m2 = df["price_per_m2"].median()
ax.axvline(median_price_per_m2, color= "red", linestyle = "--",
           label = f"Median: {median_price_per_m2}$/m2")

# let's format the edge info
ax.legend(loc="upper right", fontsize = 10, frameon = False, title = "Statistics")

# Plot the final figure
plt.tight_layout()
plt.show()

"""
 Let's read the price_per_m2 distribution  
....Right skewed**
....The median is to the left of the center which verfies the long tail to the right**
#We can use # %% to format our notes in jupyter
"""


""" Aggregate By State"""
state_price_stats = (
    df
    .groupby("state")
    ["price_per_m2"]
    .agg(['mean', 'median','std'])
    .sort_values(by = 'median', ascending = False)
    .round(2)
)
print(f"Price per m2 Statistice by state {state_price_stats}")
