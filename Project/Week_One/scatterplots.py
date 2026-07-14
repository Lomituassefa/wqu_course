import matplotlib.pyplot as plt
import pandas as pd

data_path = "/home/lomi/Desktop/wqu_courses/wqu_course/Project/mexico-real-estate-combined-clean.csv"
df = pd.read_csv(data_path)
fig, ax= plt.subplots()

ax.scatter(x = df["area_m2"], y= df["price_usd"])
ax.set_xlabel("Area_m2")
ax.set_ylabel("Price_USD")
ax.set_title("Area_vs.Price")

plt.show()