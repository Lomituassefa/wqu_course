import matplotlib.pyplot as plt
import pandas as pd


data_path = "/home/lomi/Desktop/wqu_courses/wqu_course/Project/mexico-real-estate-combined-clean.csv"
df = pd.read_csv(data_path)


lower_cut = 0.05
upper_cut = 0.90

df_clean = (
    df[
        (df["price_usd"]>df["price_usd"].quantile(lower_cut))&
        (df["price_usd"]<df["price_usd"].quantile(upper_cut))&
        (df["area_m2"]<df["area_m2"].quantile(upper_cut))

    ]
)

if __name__ == "__main__":
# To see what is changed
    print (f"Original row: {len(df)}")
    print (f"Cleaned rows: {len(df_clean)}")
    print(f"Removed rows: {len(df)- len(df_clean)}")

    # Plot 

    fig, ax = plt.subplots()

    ax.scatter(df_clean["area_m2"], df_clean["price_usd"])
    ax.set_xlabel("Area_M2")
    ax.set_ylabel("Price [USD]")
    ax.set_title("Area vs. Proce('Outliners removed')")
    plt.show()