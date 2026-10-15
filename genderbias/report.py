"""Figures + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from genderbias.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(hist, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([h["epoch"] for h in hist], [h["loss"] for h in hist], color=MULTI_SERIES_COLORS[0], linewidth=LINEWIDTH_MAIN)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE loss")
    ax.set_title("FR backbone training (CelebA)")
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_gap(fair: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    base = fair.get("baseline", {})
    names = ["Acc all", "Acc FF", "Acc MM", "Gap MM−FF"]
    vals = [
        base.get("acc_all", 0) * 100,
        base.get("acc_ff", 0) * 100,
        base.get("acc_mm", 0) * 100,
        base.get("gap_mm_minus_ff", 0) * 100,
    ]
    x = np.arange(len(names))
    ax.bar(x, vals, color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Value (×100)")
    ax.set_title("Gender gap on CelebA probe pairs")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_filters(fair: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    filters = [f for f in fair.get("vanishing_filters", []) if "filter" in f][:8]
    if not filters:
        ax.text(0.5, 0.5, "no filters", ha="center")
    else:
        labels = [f["filter"][:28] for f in filters]
        gaps = [abs(f["gap"]) * 100 for f in filters]
        x = np.arange(len(labels))
        ax.barh(x, gaps, color=MULTI_SERIES_COLORS[1], edgecolor="#002080")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("|Gender gap| (×100)")
        ax.set_title("Top attribute filters (smaller |gap| = vanishing)")
    apply_style(ax, grid_axis="x")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    fair = eval_res.get("fairness", {})
    base = fair.get("baseline", {})
    filters = [f for f in fair.get("vanishing_filters", []) if "filter" in f]
    best = filters[0] if filters else {}
    lines = [
        "# REPORT — Illusion of Gender Bias via Non-demographic Attributes",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin of Kurz et al.: decorrelated CelebA attributes, gender-gap on",
        "same/different probe pairs, and unsupervised vanishing-bias filter search.",
        "",
        "## Training",
        "",
        f"- Epochs: {train_s.get('epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- Non-demographic attrs: {train_s.get('n_nd_attrs')}",
        f"- Elapsed (s): {train_s.get('elapsed_sec', 0):.1f}",
        "",
        f"![Training]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## Verification utility",
        "",
        "| Dataset | Accuracy | EER | N |",
        "|---------|----------|-----|---|",
    ]
    for k, v in sorted(eval_res.get("verification", {}).items()):
        lines.append(f"| {k} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% | {v['n_pairs']} |")
    lines += [
        "",
        "## Gender gap (baseline)",
        "",
        f"- Acc FF: **{base.get('acc_ff', float('nan')):.4f}** (n={base.get('n_ff', 0)})",
        f"- Acc MM: **{base.get('acc_mm', float('nan')):.4f}** (n={base.get('n_mm', 0)})",
        f"- Gap (MM−FF): **{base.get('gap_mm_minus_ff', float('nan')):.4f}**",
        f"- Attr clusters (|corr|≥0.5): {fair.get('n_attr_clusters', 0)}",
        "",
        f"![Gap]({fig_dir.as_posix()}/gender_gap.svg)",
        "",
        "## Vanishing-bias filters",
        "",
    ]
    if best:
        lines += [
            f"- Best filter: `{best.get('filter')}`",
            f"- Filtered gap: **{best.get('gap', float('nan')):.4f}** (AFI={best.get('afi', float('nan')):.4f}, n={best.get('n_pairs', 0)})",
            "",
        ]
    lines += [
        "| Filter | |Gap| | AFI | N |",
        "|--------|------|-----|---|",
    ]
    for f in filters[:8]:
        lines.append(
            f"| {f['filter']} | {abs(f['gap']):.4f} | {f['afi']:.4f} | {f['n_pairs']} |"
        )
    lines += [
        "",
        f"![Filters]({fig_dir.as_posix()}/vanishing_filters.svg)",
        "",
        "## Notes",
        "",
        "- Paper uses MAAD-Face lineage; this rebuild uses CelebA 39 non-demographic attrs + Male.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def generate(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s["history"], fig_dir / "train_loss.svg")
    plot_gap(eval_res.get("fairness", {}), fig_dir / "gender_gap.svg")
    plot_filters(eval_res.get("fairness", {}), fig_dir / "vanishing_filters.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    generate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
