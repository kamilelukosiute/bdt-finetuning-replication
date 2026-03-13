"""
Plot perplexity distributions comparing Pretrained vs Finetuned Evo2 models.

Reads the CSV output from evaluate_perplexity.py and generates a boxplot + swarmplot
faceted by train/test split, matching the style of King et al. Figure S3G-H.

Usage:
    python scripts/plot_perplexity.py \
        --input results/perplexity.csv \
        --output results/perplexity_plot.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


def add_stat_annotation(ax, data, x, y, hue, pair, y_max_offset=0.15):
    """Add a statistical comparison bracket with p-value between two groups."""
    group1_label, group2_label = pair

    group1_data = data[data[hue] == group1_label][y]
    group2_data = data[data[hue] == group2_label][y]

    # Wilcoxon rank-sum test (Mann-Whitney U)
    stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative="two-sided")

    # Format p-value
    if p_value < 2.22e-16:
        p_text = "p < 2.22e-16"
    elif p_value < 0.001:
        p_text = f"p = {p_value:.2e}"
    elif p_value < 0.05:
        p_text = f"p = {p_value:.4f}"
    else:
        p_text = f"p = {p_value:.3f}"

    # Get positions for bracket
    x_categories = data[x].unique()
    x_pos = {cat: i for i, cat in enumerate(x_categories)}
    x1, x2 = x_pos[group1_label], x_pos[group2_label]

    y_max = data[y].max()
    bracket_height = y_max + y_max_offset
    bracket_top = bracket_height + 0.05

    # Draw bracket
    ax.plot(
        [x1, x1, x2, x2],
        [bracket_height, bracket_top, bracket_top, bracket_height],
        color="black",
        linewidth=1,
    )
    ax.text(
        (x1 + x2) / 2,
        bracket_top + 0.02,
        p_text,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot perplexity distributions")
    parser.add_argument("--input", default="results/perplexity.csv", help="Input CSV from evaluate_perplexity.py")
    parser.add_argument("--output", default="results/perplexity_plot.png", help="Output plot path")
    parser.add_argument("--dpi", type=int, default=300, help="Plot DPI")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")
    print(f"Models: {df['model_type'].unique()}")
    print(f"Splits: {df['split'].unique()}")
    print(f"\nPerplexity summary:")
    print(df.groupby(["split", "model_type"])["perplexity"].describe())

    # Plot setup — faceted by split (Train / Test)
    splits = ["Train", "Test"]
    available_splits = [s for s in splits if s in df["split"].values]

    fig, axes = plt.subplots(
        len(available_splits), 1,
        figsize=(6, 4 * len(available_splits)),
        sharex=True,
    )
    if len(available_splits) == 1:
        axes = [axes]

    palette = {"Pretrained": "#7AABCF", "FT-bacteriophages": "#7BC88F"}

    for ax, split in zip(axes, available_splits):
        split_df = df[df["split"] == split]

        # Boxplot
        sns.boxplot(
            data=split_df,
            x="model_type",
            y="perplexity",
            hue="model_type",
            palette=palette,
            width=0.5,
            linewidth=1.2,
            fliersize=0,
            ax=ax,
        )

        # Swarmplot (individual points)
        sns.stripplot(
            data=split_df,
            x="model_type",
            y="perplexity",
            hue="model_type",
            palette=palette,
            alpha=0.5,
            size=4,
            jitter=True,
            ax=ax,
        )

        ax.set_title(split, fontsize=14, fontweight="bold", pad=10,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#E0E0E0", edgecolor="none"))
        ax.set_xlabel("")
        ax.set_ylabel("Perplexity", fontsize=12)
        ax.tick_params(axis="x", rotation=30)

        # Add stat annotation if both model types present
        model_types = split_df["model_type"].unique()
        if "Pretrained" in model_types and "FT-bacteriophages" in model_types:
            add_stat_annotation(
                ax, split_df, x="model_type", y="perplexity", hue="model_type",
                pair=("Pretrained", "FT-bacteriophages"),
            )

    fig.suptitle(
        "Perplexity Distribution Across Models\nEvo2 Model Evaluation on Virus Sequences",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
