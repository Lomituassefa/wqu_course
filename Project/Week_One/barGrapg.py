import matplotlib.pyplot as plt
import pandas as pd


data_path = "/home/lomi/Desktop/wqu_courses/wqu_course/Project/mexico-real-estate-combined-clean.csv"
df = pd.read_csv(data_path)


mean_price_by_state = (
    df.groupby("state")["price_usd"]
    .mean()
    .sort_values(ascending = False)
)

fig, ax =plt.subplots()
mean_price_by_state.plot(
    kind = "bar",
    ax =ax
)

# Customize using Matplotlib methods
ax.set_xlabel("State")
ax.set_ylabel("Mean Price [USD]")
ax.set_title("Mean House Price by State")

plt.show()