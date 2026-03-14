"""
Parse Savanna training log and plot loss curve.

Usage:
    python scripts/plot_training_curve.py [--log logs/training.log] [--output results/training_curve.png]
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    """Extract iteration, loss, lr, throughput from Savanna log."""
    pattern = re.compile(
        r"iteration\s+(\d+)/\s*\d+.*?"
        r"learning rate:\s*([\d.E+-]+).*?"
        r"lm_loss:\s*([\d.E+-]+)"
    )
    iterations, losses, lrs = [], [], []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iterations.append(int(m.group(1)))
                lrs.append(float(m.group(2)))
                losses.append(float(m.group(3)))
    return np.array(iterations), np.array(losses), np.array(lrs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/training.log")
    parser.add_argument("--output", default="results/training_curve.png")
    args = parser.parse_args()

    iterations, losses, lrs = parse_log(args.log)
    if len(iterations) == 0:
        print("No training data found in log.")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Loss
    ax1.plot(iterations, losses, "b-", alpha=0.3, linewidth=0.5)
    if len(losses) > 20:
        window = min(50, len(losses) // 5)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax1.plot(iterations[window - 1 :], smoothed, "b-", linewidth=2, label=f"Smoothed (w={window})")
        ax1.legend()
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training Loss (iter {iterations[0]}-{iterations[-1]})")
    ax1.grid(True, alpha=0.3)

    # LR
    ax2.plot(iterations, lrs, "r-", linewidth=1)
    ax2.set_ylabel("Learning Rate")
    ax2.set_xlabel("Iteration")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output} ({len(iterations)} iterations)")


if __name__ == "__main__":
    main()
