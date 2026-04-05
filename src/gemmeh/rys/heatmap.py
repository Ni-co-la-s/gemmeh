"""
rys_heatmap.py — Generate heatmap from RYS search results CSV.

Shows the difference from baseline (delta), so green = better than base,
red = worse than base, white = same as base.

Usage:
    python heatmap.py --csv rys_results.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="rys_results.csv")
    parser.add_argument("--metric", type=str, default="acc_norm", choices=["acc", "acc_norm"])
    parser.add_argument("--output", type=str, default="rys_heatmap.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    df = df.dropna(subset=[args.metric])

    # Extract baseline
    base_row = df[df["i"] == "base"]
    if len(base_row) == 0:
        print("ERROR: No baseline row found in CSV (i=base, j=base)")
        return
    baseline = base_row[args.metric].values[0]
    print(f"Baseline {args.metric}: {baseline:.3f}")

    # Filter to only RYS configs (exclude baseline)
    df_rys = df[df["i"] != "base"].copy()
    df_rys["i"] = df_rys["i"].astype(int)
    df_rys["j"] = df_rys["j"].astype(int)

    # Build delta heatmap
    n_layers = 30
    heatmap = np.full((n_layers, n_layers), np.nan)

    for _, row in df_rys.iterrows():
        i, j = int(row["i"]), int(row["j"])
        heatmap[i, j] = row[args.metric] - baseline

    fig, ax = plt.subplots(figsize=(14, 12))

    valid = heatmap[~np.isnan(heatmap)]
    abs_max = max(abs(valid.min()), abs(valid.max()))

    # Diverging colormap centered at 0 (baseline)
    im = ax.imshow(heatmap, cmap="RdYlGn", origin="upper", vmin=-abs_max, vmax=abs_max, aspect="equal")

    ax.set_xlabel("j (end of duplicated range)", fontsize=13)
    ax.set_ylabel("i (start of duplicated range)", fontsize=13)

    ax.set_xticks(range(0, n_layers, 2))
    ax.set_yticks(range(0, n_layers, 2))

    _ = plt.colorbar(im, ax=ax, label=f"Δ {args.metric} (vs baseline {baseline:.3f})", shrink=0.8)

    # Mark the best config
    best = df_rys.loc[df_rys[args.metric].idxmax()]
    bi, bj = int(best["i"]), int(best["j"])
    best_delta = best[args.metric] - baseline

    # Mark the worst config
    worst = df_rys.loc[df_rys[args.metric].idxmin()]
    wi, wj = int(worst["i"]), int(worst["j"])
    worst_delta = worst[args.metric] - baseline

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap: {args.output}")
    print(f"Baseline: {baseline:.3f}")
    print(f"Best: i={bi}, j={bj}, {args.metric}={best[args.metric]:.3f}, Δ={best_delta:+.3f}")
    print(f"Worst: i={wi}, j={wj}, {args.metric}={worst[args.metric]:.3f}, Δ={worst_delta:+.3f}")

    # Print top 10 and bottom 10
    print("\nTop 10 configs:")
    top10 = df_rys.nlargest(10, args.metric)
    for _, row in top10.iterrows():
        delta = row[args.metric] - baseline
        print(
            f"  i={int(row['i']):2d}, j={int(row['j']):2d}, "
            f"dup={int(row['dup_size']):2d}, layers={int(row['total_layers']):2d}, "
            f"{args.metric}={row[args.metric]:.3f}, Δ={delta:+.3f}"
        )

    print("\nBottom 10 configs:")
    bot10 = df_rys.nsmallest(10, args.metric)
    for _, row in bot10.iterrows():
        delta = row[args.metric] - baseline
        print(
            f"  i={int(row['i']):2d}, j={int(row['j']):2d}, "
            f"dup={int(row['dup_size']):2d}, layers={int(row['total_layers']):2d}, "
            f"{args.metric}={row[args.metric]:.3f}, Δ={delta:+.3f}"
        )


if __name__ == "__main__":
    main()
