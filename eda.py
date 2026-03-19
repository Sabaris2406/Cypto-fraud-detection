import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ── Plotting style ────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F1923",
    "axes.facecolor":   "#1A2535",
    "axes.edgecolor":   "#2E3F55",
    "text.color":       "#E0E8F0",
    "axes.labelcolor":  "#E0E8F0",
    "xtick.color":      "#8AA0B8",
    "ytick.color":      "#8AA0B8",
    "grid.color":       "#2E3F55",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

ILLICIT_COLOR = "#EF4444"
LICIT_COLOR   = "#22C55E"
ACCENT_COLOR  = "#3B82F6"
UNKNOWN_COLOR = "#94A3B8"

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════
# 1. DATA LOADING & MERGING
# ══════════════════════════════════════════════════════════
def load_dataset(features_path: str,
                 classes_path: str,
                 edges_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge the three Elliptic CSV files."""
    print("[EDA] Loading dataset files …")

    # --- Features ---
    features = pd.read_csv(features_path, header=None)
    col_names = ["txId", "time_step"] + [f"f{i}" for i in range(1, 166)]
    features.columns = col_names
    print(f"  Features shape : {features.shape}")

    # --- Classes ---
    classes = pd.read_csv(classes_path)
    classes.columns = classes.columns.str.strip()
    print(f"  Classes shape  : {classes.shape}")
    print(f"  Class counts   :\n{classes['class'].value_counts()}\n")

    # --- Edges ---
    edges = pd.read_csv(edges_path)
    print(f"  Edges shape    : {edges.shape}")

    # --- Merge ---
    df = features.merge(classes, on="txId", how="left")
    print(f"  Merged shape   : {df.shape}\n")

    return df, edges


# ══════════════════════════════════════════════════════════
# 2. BASIC STATISTICS
# ══════════════════════════════════════════════════════════
def basic_stats(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("BASIC DATASET STATISTICS")
    print("=" * 60)
    print(f"Total transactions   : {len(df):,}")
    vc = df["class"].value_counts()
    print(f"Illicit  (class=1)   : {vc.get('1', 0):,}")
    print(f"Licit    (class=2)   : {vc.get('2', 0):,}")
    print(f"Unknown              : {vc.get('unknown', 0):,}")

    labelled = df[df["class"] != "unknown"]
    illicit  = labelled[labelled["class"] == "1"]
    licit    = labelled[labelled["class"] == "2"]
    fraud_rate = len(illicit) / len(labelled) * 100
    print(f"\nLabelled transactions: {len(labelled):,}")
    print(f"Fraud rate (labelled): {fraud_rate:.2f}%")

    print(f"\nNull counts per column (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print()


# ══════════════════════════════════════════════════════════
# 3. CLASS DISTRIBUTION PLOT
# ══════════════════════════════════════════════════════════
def plot_class_distribution(df: pd.DataFrame) -> None:
    vc = df["class"].value_counts()
    labels = ["Illicit\n(class=1)", "Licit\n(class=2)", "Unknown"]
    counts = [vc.get("1", 0), vc.get("2", 0), vc.get("unknown", 0)]
    colors = [ILLICIT_COLOR, LICIT_COLOR, UNKNOWN_COLOR]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Class Distribution — Elliptic Bitcoin Dataset",
                 color="#E0E8F0", fontsize=16, fontweight="bold", y=1.01)

    # Bar chart
    bars = axes[0].bar(labels, counts, color=colors, width=0.55,
                       edgecolor="#0F1923", linewidth=1.2)
    for bar, val in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1500,
                     f"{val:,}", ha="center", va="bottom",
                     fontsize=11, color="#E0E8F0", fontweight="bold")
    axes[0].set_title("Transaction Counts by Class",
                      color="#E0E8F0", fontsize=13)
    axes[0].set_ylabel("Count", color="#E0E8F0")
    axes[0].set_ylim(0, max(counts) * 1.15)
    axes[0].grid(axis="y")

    # Pie chart (labelled only)
    labelled_counts = [counts[0], counts[1]]
    labelled_labels = ["Illicit (2.23%)", "Licit (97.77%)"]
    axes[1].pie(labelled_counts, labels=labelled_labels,
                colors=[ILLICIT_COLOR, LICIT_COLOR],
                autopct="%1.1f%%", startangle=140,
                textprops={"color": "#E0E8F0", "fontsize": 11},
                wedgeprops={"edgecolor": "#0F1923", "linewidth": 1.5})
    axes[1].set_title("Labelled Transactions — Fraud vs Licit",
                      color="#E0E8F0", fontsize=13)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="#0F1923")
    print(f"[EDA] Saved → {path}")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════
# 4. TEMPORAL FRAUD ANALYSIS (49 time steps)
# ══════════════════════════════════════════════════════════
def plot_temporal_analysis(df: pd.DataFrame) -> None:
    labelled = df[df["class"] != "unknown"].copy()
    labelled["is_illicit"] = (labelled["class"] == "1").astype(int)

    temporal = labelled.groupby("time_step").agg(
        total=("txId", "count"),
        illicit=("is_illicit", "sum")
    ).reset_index()
    temporal["fraud_rate"] = temporal["illicit"] / temporal["total"] * 100

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("Temporal Analysis Across 49 Time Steps",
                 color="#E0E8F0", fontsize=16, fontweight="bold")

    # Total transactions per step
    axes[0].fill_between(temporal["time_step"], temporal["total"],
                         color=ACCENT_COLOR, alpha=0.6)
    axes[0].plot(temporal["time_step"], temporal["total"],
                 color=ACCENT_COLOR, linewidth=1.5)
    axes[0].set_ylabel("Total Transactions", color="#E0E8F0")
    axes[0].set_title("Transaction Volume per Time Step",
                      color="#E0E8F0", fontsize=12)
    axes[0].grid(axis="y")

    # Illicit count per step
    axes[1].fill_between(temporal["time_step"], temporal["illicit"],
                         color=ILLICIT_COLOR, alpha=0.6)
    axes[1].plot(temporal["time_step"], temporal["illicit"],
                 color=ILLICIT_COLOR, linewidth=1.5)
    axes[1].set_ylabel("Illicit Count", color="#E0E8F0")
    axes[1].set_title("Illicit Transactions per Time Step",
                      color="#E0E8F0", fontsize=12)
    axes[1].grid(axis="y")

    # Fraud rate per step
    axes[2].fill_between(temporal["time_step"], temporal["fraud_rate"],
                         color="#F59E0B", alpha=0.5)
    axes[2].plot(temporal["time_step"], temporal["fraud_rate"],
                 color="#F59E0B", linewidth=2)
    axes[2].axhline(temporal["fraud_rate"].mean(), color="white",
                    linestyle="--", linewidth=1, label="Mean fraud rate")
    axes[2].set_ylabel("Fraud Rate (%)", color="#E0E8F0")
    axes[2].set_xlabel("Time Step", color="#E0E8F0")
    axes[2].set_title("Fraud Rate (%) per Time Step",
                      color="#E0E8F0", fontsize=12)
    axes[2].legend(loc="upper right")
    axes[2].grid(axis="y")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_temporal_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="#0F1923")
    print(f"[EDA] Saved → {path}")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════
# 5. FEATURE DISTRIBUTION — ILLICIT vs LICIT
# ══════════════════════════════════════════════════════════
def plot_feature_distributions(df: pd.DataFrame,
                               features_to_plot: list[str] | None = None) -> None:
    labelled = df[df["class"] != "unknown"].copy()
    illicit  = labelled[labelled["class"] == "1"]
    licit    = labelled[labelled["class"] == "2"]

    if features_to_plot is None:
        features_to_plot = ["f1", "f2", "f3", "f4", "f5", "f6",
                            "f7", "f8", "f9", "f10", "f11", "f12"]

    n_cols = 4
    n_rows = (len(features_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle("Feature Distributions — Illicit vs Licit",
                 color="#E0E8F0", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for i, feat in enumerate(features_to_plot):
        if feat not in labelled.columns:
            axes[i].set_visible(False)
            continue
        axes[i].hist(illicit[feat].dropna(), bins=50,
                     color=ILLICIT_COLOR, alpha=0.65, density=True,
                     label="Illicit")
        axes[i].hist(licit[feat].dropna(), bins=50,
                     color=LICIT_COLOR, alpha=0.65, density=True,
                     label="Licit")
        axes[i].set_title(feat, color="#E0E8F0", fontsize=11)
        axes[i].set_xlabel("Value", color="#8AA0B8", fontsize=9)
        axes[i].set_ylabel("Density", color="#8AA0B8", fontsize=9)
        axes[i].legend(fontsize=8)
        axes[i].grid(axis="y")

    for j in range(len(features_to_plot), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="#0F1923")
    print(f"[EDA] Saved → {path}")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════
# 6. CORRELATION HEATMAP (first 30 features for readability)
# ══════════════════════════════════════════════════════════
def plot_correlation_heatmap(df: pd.DataFrame,
                             n_features: int = 30) -> None:
    labelled = df[df["class"] != "unknown"].copy()
    feat_cols = [c for c in labelled.columns
                 if c.startswith("f")][:n_features]
    corr = labelled[feat_cols].corr()

    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap,
                vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=0.3,
                annot=False, ax=ax,
                cbar_kws={"shrink": 0.7})
    ax.set_title(f"Correlation Heatmap — First {n_features} Features",
                 color="#E0E8F0", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#8AA0B8")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_correlation_heatmap.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor="#0F1923")
    print(f"[EDA] Saved → {path}")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════
# 7. BOX PLOTS — KEY FEATURES
# ══════════════════════════════════════════════════════════
def plot_boxplots(df: pd.DataFrame,
                  features: list[str] | None = None) -> None:
    labelled = df[df["class"] != "unknown"].copy()
    labelled["label"] = labelled["class"].map({"1": "Illicit", "2": "Licit"})

    if features is None:
        features = ["f1", "f2", "f3", "f4", "f5", "f6"]

    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle("Box Plots — Illicit vs Licit Distributions",
                 color="#E0E8F0", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for i, feat in enumerate(features):
        if feat not in labelled.columns:
            axes[i].set_visible(False)
            continue
        data = [labelled[labelled["label"] == "Illicit"][feat].dropna().values,
                labelled[labelled["label"] == "Licit"][feat].dropna().values]
        bp = axes[i].boxplot(data, patch_artist=True,
                             medianprops={"color": "white", "linewidth": 2},
                             whiskerprops={"color": "#8AA0B8"},
                             capprops={"color": "#8AA0B8"},
                             flierprops={"marker": "o", "markersize": 2,
                                         "alpha": 0.3})
        bp["boxes"][0].set_facecolor(ILLICIT_COLOR)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(LICIT_COLOR)
        bp["boxes"][1].set_alpha(0.7)
        axes[i].set_xticklabels(["Illicit", "Licit"])
        axes[i].set_title(feat, color="#E0E8F0", fontsize=11)
        axes[i].grid(axis="y")

    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="#0F1923")
    print(f"[EDA] Saved → {path}")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════
# 8. SUMMARY STATISTICS TABLE
# ══════════════════════════════════════════════════════════
def print_summary_table(df: pd.DataFrame) -> None:
    labelled = df[df["class"] != "unknown"].copy()
    illicit = labelled[labelled["class"] == "1"]
    licit   = labelled[labelled["class"] == "2"]

    feat_cols = [c for c in labelled.columns if c.startswith("f")][:10]

    print("\n" + "=" * 70)
    print("FEATURE SUMMARY — ILLICIT vs LICIT (first 10 features)")
    print("=" * 70)
    print(f"{'Feature':<8} {'Illicit Mean':>14} {'Licit Mean':>12} "
          f"{'Illicit Std':>13} {'Licit Std':>11}")
    print("-" * 70)
    for feat in feat_cols:
        print(f"{feat:<8} {illicit[feat].mean():>14.4f} "
              f"{licit[feat].mean():>12.4f} "
              f"{illicit[feat].std():>13.4f} "
              f"{licit[feat].std():>11.4f}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def run_eda(features_path: str = "elliptic_txs_features.csv",
            classes_path: str  = "elliptic_txs_classes.csv",
            edges_path: str    = "elliptic_txs_edgelist.csv") -> pd.DataFrame:
    """Run the complete EDA pipeline and return the merged DataFrame."""

    df, edges = load_dataset(features_path, classes_path, edges_path)
    basic_stats(df)
    plot_class_distribution(df)
    plot_temporal_analysis(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    print_summary_table(df)

    print("\n[EDA] All plots saved to:", OUTPUT_DIR)
    return df


if __name__ == "__main__":
    run_eda()
