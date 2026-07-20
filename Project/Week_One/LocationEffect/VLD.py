""" Visualizing Location differnces"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df =(
    pd.read_csv("/home/lomi/Desktop/wqu_courses/wqu_course/Project/mexico-real-estate-combined-clean.csv")
    .assign(price_per_m2 = lambda x: x["price_usd"]/x["area_m2"])
)
""" Aggregate By State"""
state_price_stats = (
    df
    .groupby("state")
    ["price_per_m2"]
    .agg(['mean', 'median','std'])
    .sort_values(by = 'median', ascending = False)
    .round(2)
)
# print(f"Price per m2 Statistice by state {state_price_stats}")

# Identifying the states we plot
top_states = state_price_stats.head(3).index.tolist()
bottom_states = state_price_stats.tail(3).index.tolist()
target_states =  top_states + bottom_states 

# Filter data by target_states

df_states = df[df["state"].isin(target_states)]

# Ready for plotting 
data_to_plot = [df_states[df_states["state"] ==state]["price_per_m2"]for state in target_states]
# Plotting

fig, ax = plt.subplots()
ax.boxplot(data_to_plot, tick_labels=target_states, orientation = 'vertical')
ax.set_title("Price per m² Distribution: Most vs. Least Expensive States")
ax.set_xlabel("Price per m² [USD]")
ax.set_ylabel("State")

plt.tight_layout()
plt.show()


"""Quantifying Location vs Size Effects: Variance Decomposition
Interpretation:If between state variance >>within state variance, then knowing which state 
a property is in explain more of its price than any within-state factors (like size)
. this directly answets our researches question"""

# 1. Correlation between size and price
size_price_correlation = df[["area_m2", "price_usd"]].corr().iloc[0, 1]
print(f"Correlation between area and price: {size_price_correlation:.3f}")

# 2. Calculate variance components
# Total variance in price_usd
total_variance = df["price_usd"].var()

# Between-state variance (variance of state_means)
state_means = df.groupby("state")["price_usd"].mean()
between_variance = state_means.var()

# Within-state variance (average variance within each state)
within_variance = df.groupby("state")["price_usd"].var().mean()

# 3. Calculate percentages
between_pct = (between_variance / total_variance) * 100
within_pct = (within_variance / total_variance) * 100

print(f"\nVariance Decomposition:")
print(f"  Between-state variance: {between_pct:.1f}%")
print(f"  Within-state variance: {within_pct:.1f}%")
print(f"  Ratio (Between/Within): {between_variance/within_variance:.2f}")
