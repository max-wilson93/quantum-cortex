import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze_interactions():
    filename = "monte_carlo_physics.csv"
    if not os.path.exists(filename):
        print("Error: CSV file not found.")
        return

    print(f"-> Loading {filename}...")
    df = pd.read_csv(filename)

    # 1. CLEANING DATA
    # Remove '%' and 's' and convert to float
    if df['Accuracy'].dtype == 'O':
        df['Accuracy'] = df['Accuracy'].str.replace('%', '').astype(float)
    if df['Duration'].dtype == 'O':
        df['Duration'] = df['Duration'].str.replace('s', '').astype(float)

    # 2. CREATE "PERFORMANCE TIERS"
    # We create a category to color the dots nicely
    # 'Elite' = Top 10%, 'Good' = Top 50%, 'Bad' = Bottom 50%
    p90 = df['Accuracy'].quantile(0.90)
    p50 = df['Accuracy'].quantile(0.50)

    def get_tier(acc):
        if acc >= p90: return 'Elite'
        if acc >= p50: return 'Good'
        return 'Bad'

    df['Tier'] = df['Accuracy'].apply(get_tier)

    print(f"-> Data Loaded. Elite Threshold (>90%): {p90:.2f}%")

    # 3. THE PAIR PLOT (All vs All)
    # We exclude Run_ID and Duration to focus on Physics
    physics_vars = ['Accuracy', 'LR', 'Flex', 'Lat_Str', 'Thresh', 'Kerr', 'Sys_E']
    
    print("-> Generating Phase Space PairPlot (This may take a moment)...")
    plt.figure(figsize=(20, 20))
    
    # Use a color palette where Elite is Bright Red/Orange
    pair_plot = sns.pairplot(
        df, 
        vars=physics_vars, 
        hue='Tier', 
        hue_order=['Bad', 'Good', 'Elite'],
        palette={'Bad': 'gray', 'Good': 'blue', 'Elite': 'red'},
        plot_kws={'alpha': 0.6, 's': 20},
        diag_kind='kde' # Shows the distribution shape
    )
    
    plt.subplots_adjust(top=0.95)
    pair_plot.fig.suptitle('Quantum Phase Space Analysis: Finding the Sweet Spot')
    plt.savefig("analysis_pairplot.png")
    print("-> Saved 'analysis_pairplot.png'")

    # 4. THE CORRELATION HEATMAP (The Math)
    plt.figure(figsize=(10, 8))
    # Only correlate numeric columns
    corr = df[physics_vars].corr()
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Linear Correlation Matrix")
    plt.savefig("analysis_heatmap.png")
    print("-> Saved 'analysis_heatmap.png'")
    
    # 5. PRINT ELITE STATS
    elite_df = df[df['Tier'] == 'Elite']
    print("\n=== ELITE CONFIGURATION RANGES (Top 10%) ===")
    print(elite_df[physics_vars].describe().loc[['mean', 'min', 'max', 'std']])

if __name__ == "__main__":
    analyze_interactions()