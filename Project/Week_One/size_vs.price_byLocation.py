from scatterplots_QuantileFiltering import df_clean
import matplotlib.pyplot as plt
import seaborn as sns

top_states =  df_clean["state"].value_counts().head(3).index.tolist()

fig, axes = plt.subplots(3, 1, figsize=(5, 18), sharey=True)

df_states = df_clean[df_clean["state"].isin(top_states)]

# 4. Loop through the states and plot on each specific axis
states = df_states["state"].unique()

for ax, state in zip(axes, states):
    # Filter data for the current state
    subset = df_states[df_states["state"] == state]
    
    # Use regplot (Axes-level) which ACCEPTS 'ax='
    sns.regplot(
        data=subset,
        x= "area_m2" , # <--- "area_m2"
        y="price_usd",  # <--- "price_usd" 
        color='tab:blue', # Optional: keep colors consistent
        ax=ax # <--- This is where you pass the axis explicitly
    )
    
    ax.set_title(f"State: {state}")

plt.tight_layout(pad = 3.0)
plt.subplots_adjust(hspace=0.4)
plt.show()