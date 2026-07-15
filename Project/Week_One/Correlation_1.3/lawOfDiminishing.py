import matplotlib.pyplot as plt
import pandas as pd
from pearsonCorrelation import df

fig, ax = plt.subplots()

# Scatter plot
ax.scatter(x=df["area_m2"], y=df["price_per_m2"],)

ax.set_title("Are larger properties cheaper per unit?")
ax.set_xlabel("Area [sq meters]")
ax.set_ylabel("Price per m2 [USD/m2]")

plt.show()