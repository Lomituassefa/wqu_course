# Histogram of price_usd

import matplotlib.pyplot as plt
import pandas as pd
data_path = "/home/lomi/Desktop/wqu_courses/wqu_course/Project/mexico-real-estate-combined-clean.csv"
df = pd.read_csv(data_path)
# print(f"Dimenstions: {df.shape[0]} rows and {df.shape[1]} columns.")
# print(df.info())

# fig, ax= plt.subplots()
# ax.hist(df["price_usd"])
# ax.set_xlabel("Price [USD]")
# ax.set_ylabel("Frequency")
# ax.set_title("Distribution of Property Price")
# plt.show()

# fig, ax= plt.subplots()
# ax.hist(df["area_m2"])
# ax.set_xlabel("Area_m2")
# ax.set_ylabel("Frequency")
# ax.set_title("Distribution of Property Price")
# plt.show()

# prop_type = df["property_type"].unique()
# data_by_type = [df[df["property_type"]==label]["price_usd"] for label in prop_type]
# # print(data_by_type)
# fig, ax = plt.subplots()
# ax.boxplot(data_by_type, tick_labels = prop_type, orientation='horizontal')   # we can also use vert = False
# ax.set_title("Price Distribution by Property type")
# ax.set_xlabel("Price[USD]")
# plt.tight_layout()
# plt.show()

selected_state = (
    df["state"]
    .value_counts()
    .head(3)
    .index.tolist()
)
data_by_state = [df[df["state"]==state]["price_usd"] for state in selected_state]
fig, ax = plt.subplots()
ax.boxplot(data_by_state, tick_labels= selected_state, orientation= 'horizontal')

ax.set_title("Price Distribution by State")
ax.set_xlabel("Price [USD]")
plt.tight_layout()
plt.show()