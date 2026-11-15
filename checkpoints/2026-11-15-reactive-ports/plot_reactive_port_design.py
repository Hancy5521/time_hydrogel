"""Plot a witnessed schedule pair and the frozen policy comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "scores", "diagnostics", "out"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    protocol, scores, diagnostics = [json.loads(getattr(args, item).read_text(encoding="utf-8"))
                                     for item in ("protocol", "scores", "diagnostics")]
    witness = diagnostics["first_eligible_case"]
    goal = protocol["goal"]
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), layout="constrained", gridspec_kw={"width_ratios": (1, 1, 1.45)})
    for i, label in enumerate(("A    Region A first", "B    Region B first")):
        axis = axes[i]
        for region, color in enumerate(("#087c70", "#b06332")):
            y = np.asarray(witness["regional_curves"])[i, :, region]
            axis.plot(witness["times"], y, color=color, linewidth=2.2, label=f"Region {'AB'[region]}")
            axis.plot(witness["crossings"][i][region], goal["threshold"], "o", color=color, ms=4)
        axis.axhline(goal["threshold"], color="#758498", linestyle="--", linewidth=1)
        axis.vlines(goal["deadline"], *goal["terminal_window"], color="#a0c8bd", alpha=.80, linewidth=7)
        item = witness["actions"][i]
        axis.axvline(protocol["prefix_duration"] + protocol["feed_duration"] * item["early_fraction"], color="#b7ada0", lw=1)
        axis.text(.05, .91, f"Split: {item['early_split']:+.2f} then {item['late_split']:+.2f}",
                  transform=axis.transAxes, fontsize=9)
        axis.set(xlabel="Model time", title=label, ylim=(0, 1), xlim=(.58, 4.42))
    axes[0].set_ylabel("Regional crosslink fraction")
    axes[0].legend(frameon=False, loc="lower right")
    compare = scores["groups"]
    method = ("fixed", "two_point_sign", "bound_only", "free_only", "direct_quadratic", "local_joint")
    labels = ("Fixed", "Two-point sign", "Two bound reads", "Two free reads", "Direct quadratic", "Four reads, physics")
    axis = axes[2]
    y = np.arange(len(labels))
    for j, group in enumerate(compare):
        vals = [group["policies"][m]["success"] * 100 for m in method]
        axis.barh(y + (-.15 if j == 0 else .15), vals, color=("#087c70", "#7a8aa0")[j], height=.29,
                  label=("New configurations", "Shifted kinetics")[j])
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set(title="C    Held-out synthetic control", xlabel="Joint success (%)", xlim=(0, 53))
    axis.legend(frameon=False, fontsize=8, loc="upper right")
    for axis in axes:
        axis.grid(alpha=.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(args.out, dpi=170)


if __name__ == "__main__":
    main()
