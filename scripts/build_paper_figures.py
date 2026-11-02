"""Render the manuscript's three figures from generated result artefacts.

All are built for ACM two-column print, so the binding constraint is greyscale
legibility rather than screen colour: every distinction that carries meaning is
also carried by lightness, position, or a direct label.

Figures 1 and 2 are schematics that encode no data; they use ink, one accent
hue, and plain grey bands. Figure 3 uses the emphasis form: one accent for the
single-choice bars, a darker step of the same hue for the compound bar, and a
de-emphasis grey with a thin dark edge for the low-turnover controls, each drawn
directly beneath the cell it pairs with. Fills are solid throughout, TrueType
fonts are embedded, and every bar carries its value label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
ACCENT = "#2a78d6"
CONTROL = "#a8a7a2"
SINGLE_FILL = "#6da7ec"     # faint blue for the single-choice bars
COMPOUND_FILL = ACCENT      # the compound bar takes the stronger mid-blue
HAIRLINE = "#d8d7d3"

plt.rcParams.update({
    # Type 42 embeds TrueType outlines. Matplotlib defaults to Type 3, which
    # renders fuzzily in some viewers, leaves figure text unsearchable, and is
    # flagged by ACM production checks at camera-ready.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "Linux Libertine", "DejaVu Serif"],
    "font.size": 7.2,
    "axes.edgecolor": HAIRLINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def figure_timeline(output: Path) -> None:
    """Schematic of the causal decision timeline and the leakage each control closes."""
    fig, ax = plt.subplots(figsize=(7.0, 1.95))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.1)
    ax.axis("off")

    ax.annotate("", xy=(9.7, 1.5), xytext=(0.3, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=INK_SOFT, linewidth=1.0))

    stops = [
        (1.5, "month-end $t$", "eligible universe\nand prices through $t$"),
        (4.0, "signal (at $t$)", "graphs, regime state,\nweights fitted on $<t$ only"),
        (6.5, "$t{+}1$ close", "execution;\ncosts deducted"),
        (9.0, "$>t{+}1$", "realised return,\nlater training label only"),
    ]
    for x, label, detail in stops:
        ax.plot([x], [1.5], marker="o", markersize=5.5, color=ACCENT,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
        ax.text(x, 1.92, label, ha="center", va="bottom", fontsize=7.6, color=INK, weight="bold")
        ax.text(x, 1.24, detail, ha="center", va="top", fontsize=6.6, color=INK_SOFT, linespacing=1.35)

    barrier = 5.25
    ax.plot([barrier, barrier], [0.45, 2.75], linestyle="-", linewidth=1.1, color=INK)
    ax.text(barrier - 0.12, 2.82, "information barrier", ha="right", va="bottom",
            fontsize=6.8, color=INK, style="italic")

    ax.add_patch(Rectangle((0.3, 0.45), barrier - 0.3, 0.42, facecolor=ACCENT,
                           alpha=0.10, edgecolor="none"))
    ax.text((0.3 + barrier) / 2, 0.66, "observable at decision time",
            ha="center", va="center", fontsize=6.8, color=INK)
    # Plain fill: a hatch under the text made it harder to read. The darker grey
    # alone separates the two regions, in colour and in greyscale.
    ax.add_patch(Rectangle((barrier, 0.45), 9.7 - barrier, 0.42, facecolor="#e2e1dd",
                           edgecolor="none"))
    ax.text((barrier + 9.7) / 2, 0.66, "may never influence a weight, scaler, or hyperparameter",
            ha="center", va="center", fontsize=6.8, color=INK)

    fig.tight_layout(pad=0.3)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_pipeline(output: Path) -> None:
    """Schematic of the release path: what must hold before a run becomes a claim.

    A diagram, not a chart -- it encodes no data. The point it carries is that
    each stage can only *refuse*: nothing here makes a result better, so the
    whole path is a sequence of gates rather than a processing chain.
    """
    fig, ax = plt.subplots(figsize=(7.0, 1.72))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    stages = [
        ("Data\ncontract", "universe, adjustment,\nsource hashes"),
        ("Canonical\nrun", "config, inputs, seed,\ncode revision"),
        ("Candidate\nledger", "family frozen\nbefore results"),
        ("Multiplicity-aware\ninference", "paired bootstrap,\nReality Check, DSR"),
        ("Release\ngate", "five evidence flags,\nnon-paper by default"),
    ]
    width, gap = 16.0, 4.6
    for i, (title, detail) in enumerate(stages):
        x = 1.5 + i * (width + gap)
        ax.add_patch(Rectangle((x, 11.5), width, 11.5, facecolor=SURFACE,
                               edgecolor=ACCENT, linewidth=1.0))
        ax.text(x + width / 2, 19.6, title, ha="center", va="center",
                fontsize=7.4, color=INK, weight="bold", linespacing=1.25)
        ax.text(x + width / 2, 14.4, detail, ha="center", va="center",
                fontsize=6.3, color=INK_SOFT, linespacing=1.3)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + width + gap - 0.7, 17.2), xytext=(x + width + 0.5, 17.2),
                        arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.1))
        # Every stage fails closed; the refusal is the mechanism, not a fallback.
        ax.annotate("", xy=(x + width / 2, 6.4), xytext=(x + width / 2, 11.2),
                    arrowprops=dict(arrowstyle="-|>", color=CONTROL, linewidth=0.8,
                                    linestyle=(0, (3, 2))))

    ax.add_patch(Rectangle((1.5, 2.6), 5 * width + 4 * gap, 3.8, facecolor="#e2e1dd",
                           edgecolor="none"))
    ax.text(50, 4.5, "any stage may refuse, and a blocked run is recorded as a result rather than bypassed",
            ha="center", va="center", fontsize=6.8, color=INK)

    fig.tight_layout(pad=0.25)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_inflation(output: Path) -> None:
    """Inflation per evaluation choice, with each control beside its treatment.

    The treatment-versus-control contrast is the evidence, so the two gross-cost
    controls sit directly beneath the cells they pair with rather than at their
    magnitude rank. The compound reading is a sum of three choices, not another
    single price, so it sits below a dashed rule instead of inside the ranking.
    Solid fills only; the compound bar is a darker step of the same hue, and the
    near-zero control bars carry a thin dark edge so they stay visible.
    """
    payload = json.loads((PROJECT_ROOT / "results" / "protocol_inflation_table_v1.json").read_text())
    by_id = {
        row["row_id"]: row
        for group in ("rows", "benchmark_choice_rows", "period_mismatch_rows")
        for row in payload.get(group, [])
    }

    # Top-to-bottom layout: singles by magnitude, each control under its pair,
    # compound last below the divider.
    ORDER = [
        ("best-seed-us50", "Best seed (50-ETF)", "single"),
        ("period-mismatch-quarterly-trackb", "Benchmark on a different period", "single"),
        ("development-window-as-held-out-20-epochs", "Development window, declared budget", "single"),
        ("costless-execution-dynamic", "Gross costs, 56.0%/mo turnover", "single"),
        ("costless-execution-equal-weight-control", "control: equal weight, 2.6%/mo", "control"),
        ("costless-execution-dynamic-trackb", "Gross costs, 48.9%/mo turnover \u2020", "single"),
        ("costless-execution-equal-weight-trackb", "control: equal weight, 2.0%/mo \u2020", "control"),
        ("development-window-as-held-out", "Development window, 5 epochs", "single"),
        ("best-seed-trackb-2020", "Best seed (confirmation, 2020+) \u2020", "single"),
        ("benchmark-choice-trackb", "One favourite baseline, not a ladder \u2020", "single"),
        ("best-seed-us30", "Best seed (30-ETF)", "single"),
        ("compound-three-shortcuts", "Three choices combined", "compound"),
    ]
    # A row added to the table without a matching entry here must fail the build,
    # not silently vanish from the figure. That has happened twice.
    missing = set(by_id) - {rid for rid, _, _ in ORDER}
    extra = {rid for rid, _, _ in ORDER} - set(by_id)
    if missing or extra:
        raise ValueError(f"figure order out of sync with table: missing {missing}, extra {extra}")

    fig, ax = plt.subplots(figsize=(7.0, 2.85))
    ys, y = [], 0.0
    for rid, _, role in ORDER:
        if role == "compound":
            y += 0.8  # the divider lives in this gap
        ys.append(y)
        y += 1.0
    divider_y = ys[-1] - 0.9

    FILL = {"single": SINGLE_FILL, "control": CONTROL, "compound": COMPOUND_FILL}
    for (rid, _, role), ypos in zip(ORDER, ys):
        value = by_id[rid]["inflation"]
        ax.barh(ypos, value, height=0.62, color=FILL[role],
                edgecolor="#555555" if role == "control" else "none",
                linewidth=0.8 if role == "control" else 0.0, zorder=2)
        ax.text(value + 0.008, ypos, f"{value:+.3f}", va="center", ha="left",
                fontsize=6.8, color=INK)

    ax.axhline(divider_y, color=CONTROL, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.invert_yaxis()

    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label, _ in ORDER], fontsize=7.0)
    for tick, (_, _, role) in zip(ax.get_yticklabels(), ORDER):
        if role == "control":
            tick.set_color(INK_SOFT)
            tick.set_fontsize(6.2)
    values = [by_id[rid]["inflation"] for rid, _, _ in ORDER]
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_xlabel("Sharpe inflation (reported $-$ adjusted)", fontsize=7.0)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(HAIRLINE)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout(pad=0.3)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_timeline(output_dir / "timeline.pdf")
    figure_pipeline(output_dir / "pipeline.pdf")
    figure_pipeline(output_dir / "pipeline.png")
    figure_inflation(output_dir / "inflation.pdf")
    figure_timeline(output_dir / "timeline.png")
    figure_inflation(output_dir / "inflation.png")
    print(f"wrote {args.output_dir}/timeline.pdf and inflation.pdf (+ .png previews)")


if __name__ == "__main__":
    main()
