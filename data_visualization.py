"""
Utility for Data Visualization for Grampa DB.
"""
from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd


def read_grampa_csv(grampa_path):
    """Read CSV using pandas."""

    df = pd.read_csv(grampa_path)
    return df


def get_dataset_overview(df):
    """-------------------------------
    # 1. Quick overview of the dataset
    # -------------------------------"""
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:\n", df.head())
    print("\nColumn types:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isna().sum())


def basic_statistics(df):
    """-------------------------------
    # 2. Basic statistics for numeric columns
    # -------------------------------"""
    # Assuming 'value' is numeric
    print("\nBasic statistics for numeric columns:")
    print(df["value"].describe())


def count_categorical_col_frequency(df):
    """-------------------------------
    # 3. Frequency counts for categorical columns
    # -------------------------------"""
    categorical_cols = [
        "bacterium",
        "modifications",
        "strain",
        "unit",
        "is_modified",
        "has_unusual_modification",
        "has_cterminal_amidation",
        "datasource_has_modifications",
        "database",
    ]

    for col in categorical_cols:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts())

        print("\nTop 5 most frequent:")
        print(df[col].value_counts().head())


def get_modification_percent(df):
    """-------------------------------
    # 4. Percentage of modified vs non-modified sequences
    # -------------------------------"""
    if "is_modified" in df.columns:
        print("\nPercentage of modified sequences:")
        print(df["is_modified"].value_counts(normalize=True) * 100)


def seq_len_analysis(df):
    """-------------------------------
    # 5. Sequence length analysis
    # -------------------------------"""
    if "sequence" in df.columns:
        df["seq_length"] = df["sequence"].apply(len)
        print("\nStatistics for sequence lengths:")
        print(df["seq_length"].describe())


def plot_seq_length_histogram(df, fig_name="seq_len_histogram.png"):
    """
    Plot Sequence Lenght Histogram
    """
    plt.figure(figsize=(10, 6))  # larger figure
    plt.hist(df["seq_length"], bins=30, color="skyblue", edgecolor="black", alpha=0.8)

    plt.xlabel("Sequence Length (aa)", fontsize=12)
    plt.ylabel("Number of Sequences", fontsize=12)
    plt.title("Distribution of Sequence Lengths", fontsize=14, fontweight="bold")

    plt.grid(axis="y", linestyle="--", alpha=0.7)  # horizontal gridlines
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()  # avoid clipping labels
    plt.savefig(f"visualization/{fig_name}", dpi=300)
    plt.show()


def compute_modification_stats(df, save_path="seq_modification_stat.png"):
    """
    Compute peptide modification statistics
    """
    if "modifications" in df.columns and "seq_length" in df.columns:
        # Correct function to count modifications
        def count_modifications(mod_str):
            if mod_str in ("[]", None, "") or pd.isna(mod_str):
                return 0
            mod_str = mod_str.strip("[]")  # Remove brackets
            mods = [
                m.strip() for m in mod_str.split(",") if m.strip()
            ]  # Split by comma
            return len(mods)

        # Apply function
        df["num_modifications"] = df["modifications"].apply(count_modifications)

        print("\n=== Modifications per sequence ===")
        print(df["num_modifications"].describe())

        # Scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df["seq_length"], df["num_modifications"], alpha=0.5)
        plt.xlabel("Sequence Length", fontsize=12)
        plt.ylabel("Number of Modifications", fontsize=12)
        plt.title(
            "Sequence Length vs Number of Modifications", fontsize=14, fontweight="bold"
        )

        plt.grid(axis="y", linestyle="--", alpha=0.7)  # horizontal gridlines
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()  # avoid clipping labels

        plt.savefig(f"visualization/{save_path}")
        plt.show()


def plot_modification_histogram(df, save_path="seq_modifcation_bar.png"):
    """Count how many sequences have 0, 1, 2,... modifications"""
    mod_dist = df["num_modifications"].value_counts().sort_index()

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(mod_dist.index, mod_dist.values, color="skyblue", edgecolor="black")
    plt.barh(mod_dist.index, mod_dist.values, color="skyblue", edgecolor="black")
    plt.ylabel("Number of Modifications", fontsize=12)
    plt.xlabel("Number of Sequences", fontsize=12)
    plt.title(
        "Distribution of Modifications per Sequence", fontsize=14, fontweight="bold"
    )
    plt.xticks(fontsize=10)  # Show each integer modification count on x-axis
    plt.yticks(mod_dist.index, fontsize=10)
    plt.gca().invert_yaxis()  # show 0 at top
    plt.tight_layout()

    plt.savefig(f"visualization/{save_path}")

    plt.show()


def perform_grampa_data_analysis(csv_path):
    """Wrapper for grampa database analysis"""
    df = read_grampa_csv(csv_path)
    get_dataset_overview(df)
    basic_statistics(df)
    count_categorical_col_frequency(df)
    get_modification_percent(df)
    seq_len_analysis(df)
    plot_seq_length_histogram(df)
    compute_modification_stats(df)
    plot_modification_histogram(df)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def plot_correlation(y_true, y_pred, save_path=None, title="Predicted vs Actual"):
    plt.figure(figsize=(12, 5))
    
    # --- Plot 1: Scatter Plot ---
    plt.subplot(1, 2, 1)
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.4, 's':10}, line_kws={'color':'red'})
    plt.xlabel("Actual ")
    plt.ylabel("Predicted")
    plt.title(f"{title}\nPearson: {pearsonr(y_true, y_pred)[0]:.4f}")
    
    # --- Plot 2: Residual Plot ---
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, color='purple')
    plt.xlabel("Prediction Error (Residual)")
    plt.title("Error Distribution")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.savefig("visualization/pearson/correlation_analysis.png")
        print("Plot saved successfully to Path: visualization/pearson/correlation_analysis.png")

    plt.show()



def plot_train_val_with_pearson(history, save_path):
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(10, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict["loss"], label="train_loss")
    plt.plot(epochs, history_dict["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Plot Val Pearson
    plt.subplot(1, 2, 2)
    if "val_Pearson" in history_dict:
        plt.plot(epochs, history_dict["val_Pearson"], label="val_Pearson", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Pearson r")
    plt.title("Validation Pearson")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()    

def plot_uncertainty_vs_error(y_true, q25, q50, q75, save_path=None):
    uncertainty = q75 - q25
    abs_error = np.abs(y_true - q50)

    rho, p = spearmanr(uncertainty, abs_error)

    plt.figure(figsize=(6, 5))
    plt.scatter(uncertainty, abs_error, alpha=0.6)
    plt.xlabel("Predictive Uncertainty (Q75 − Q25)")
    plt.ylabel("Absolute Error |y − Q50|")
    plt.title(f"Uncertainty vs Error (Spearman ρ = {rho:.2f})")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_mean_attention(attn_weights, save_path):
    labels = ["Sequence", "GNN", "Physicochemical"]
    mean_attn = attn_weights.mean(axis=0)

    plt.figure(figsize=(5, 4))
    sns.barplot(x=labels, y=mean_attn)
    plt.ylabel("Mean Attention Weight")
    plt.title("Average Branch Attention")
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    print(f'Plot saved sucessfully to Path: {save_path}')

    plt.show()

def plot_attention_distribution(attn_weights, save_path):
    labels = ["Sequence", "GNN", "Physicochemical"]

    plt.figure(figsize=(6, 4))
    for i in range(3):
        sns.kdeplot(attn_weights[:, i], label=labels[i], fill=True)

    plt.xlabel("Attention Weight")
    plt.ylabel("Density")
    plt.title("Distribution of Attention Weights Across Samples")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    print(f'Plot saved sucessfully to Path: {save_path}')

    plt.show()


def format_metrics_table(metrics_list, float_fmt="{:.4f}"):
    """
    Convert a list of metric dictionaries into a formatted table string.

    Args:
        metrics_list (list[dict]): List of metric dicts
        float_fmt (str): Float format string

    Returns:
        str: Formatted table
    """
    if not metrics_list:
        return "No metrics available."

    # Column order from first entry
    headers = ["Run"] + list(metrics_list[0].keys())

    # Prepare rows
    rows = []
    for i, metrics in enumerate(metrics_list, start=1):
        row = [str(i)]
        for key in headers[1:]:
            val = metrics.get(key, "")
            if isinstance(val, float):
                row.append(float_fmt.format(val))
            else:
                row.append(str(val))
        rows.append(row)

    # Compute column widths
    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    # Build table
    sep = " | "
    header_line = sep.join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    divider = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    body_lines = [
        sep.join(row[i].ljust(col_widths[i]) for i in range(len(headers)))
        for row in rows
    ]

    return "\n".join([header_line, divider] + body_lines)

def save_p_aeruginosa_dataset(grampa_csv_path, save_path="data/p_aeruginosa_dataset.csv"):
    df = read_grampa_csv(grampa_csv_path)
    p_aur_df = df[df["bacterium"] == "P. aeruginosa"]
    p_aur_df.to_csv(save_path, index=False)
    print(f"P. aeruginosa dataset extracted from {grampa_csv_path}")
    print(f"Total sequences for P. aeruginosa: {len(p_aur_df)}")
    print(f"P. aeruginosa dataset saved to {save_path}")
    



if __name__ == "__main__":
    P_AERUGINOSA_CSV = "data/p_aeruginosa_dataset.csv"
    GRAMPA_CSV  = "data/grampa.csv"
    ECOLI_CSV  = "data/ecoli_normalized.csv"
    # perform_grampa_data_analysis(ECOLI_CSV)
    # save_p_aeruginosa_dataset(GRAMPA_CSV, P_AERUGINOSA_CSV)