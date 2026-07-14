from scatterplots_QuantileFiltering import df_clean
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
sns.regplot(
    data = df_clean,
    x = "area_m2",
    y = "price_usd",
    line_kws = {"color": "red"},
    ax = ax
)

ax.set_xlabel("Area [sq meters]")
ax.set_ylabel("Price [USD]")
ax.set_title("Area vs. Price with Trendline")
plt.show()