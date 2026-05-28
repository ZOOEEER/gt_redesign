from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from scipy import stats


# =========================
# Global parameters
# =========================
INPUT_DIR = Path("UGT84A56")
OUTPUT_DIR = Path("output-ugt")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
FONT_FAMILY = "Arial"
FIG_WIDE = (8.0, 4.8)
FIG_SCATTER = (4.6, 4.6)
FIG_BAR_SANKY = (12.8, 4.8)
FIG_MMPBSA = (13.2, 8.6)
FIG_VIOLIN_16 = (10, 6.6)
FIG_VIOLIN_8 = (6, 6.2)

LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 9
ANNOT_SIZE = 8
SANKEY_LABEL_FONT = 7.5
SANKEY_LABEL_GAP = 1

WT_COLOR = "#8F8F8F"
SDU6_COLOR = "#1E9EFF"
SDT5_COLOR = "#D95C5C"
SDT8_COLOR = "#8C1D1D"

SYSTEM_COLORS = {
    "wt": WT_COLOR,
    "sdu6": SDU6_COLOR,
}

HYDROPHOBIC_COL = (0.92, 0.78, 0.18)
POSITIVE_COL = (0.28, 0.48, 0.88)
NEGATIVE_COL = (0.88, 0.30, 0.30)
PRO_COL = (0.88, 0.55, 0.18)
POLAR_COL = (0.58, 0.58, 0.58)

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [FONT_FAMILY, "Arial", "DejaVu Sans"]
mpl.rcParams["axes.linewidth"] = 1.0
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

AA_GROUP_ORDER = ["hydrophobic", "positive", "negative", "pro", "gly", "polar_neutral"]
AA_TO_GROUP = {
    "ALA": "hydrophobic",
    "CYS": "hydrophobic",
    "ILE": "hydrophobic",
    "LEU": "hydrophobic",
    "MET": "hydrophobic",
    "PHE": "hydrophobic",
    "TRP": "hydrophobic",
    "TYR": "hydrophobic",
    "VAL": "hydrophobic",
    "ARG": "positive",
    "HIS": "positive",
    "LYS": "positive",
    "ASP": "negative",
    "GLU": "negative",
    "PRO": "pro",
    "GLY": "gly",
    "ASN": "polar_neutral",
    "GLN": "polar_neutral",
    "SER": "polar_neutral",
    "THR": "polar_neutral",
}
AA_ORDER_SANKEY = []
for _group in AA_GROUP_ORDER:
    AA_ORDER_SANKEY.extend(sorted([aa for aa, grp in AA_TO_GROUP.items() if grp == _group]))

RESIDUE_CLASS = {
    "ALA": "hydrophobic",
    "VAL": "hydrophobic",
    "ILE": "hydrophobic",
    "LEU": "hydrophobic",
    "MET": "hydrophobic",
    "PHE": "hydrophobic",
    "TRP": "hydrophobic",
    "TYR": "hydrophobic",
    "CYS": "hydrophobic",
    "ARG": "positive",
    "LYS": "positive",
    "HIS": "positive",
    "HID": "positive",
    "HIE": "positive",
    "HIP": "positive",
    "HISD": "positive",
    "ASP": "negative",
    "GLU": "negative",
    "PRO": "pro_special",
    "GLY": "pro_special",
    "ASN": "polar",
    "GLN": "polar",
    "SER": "polar",
    "THR": "polar",
}

CLASS_COLORS = {
    "hydrophobic": HYDROPHOBIC_COL,
    "positive": POSITIVE_COL,
    "negative": NEGATIVE_COL,
    "pro_special": PRO_COL,
    "polar": POLAR_COL,
}

CLASS_MARKERS = {
    "hydrophobic": "o",
    "positive": "^",
    "negative": "s",
    "pro_special": "D",
    "polar": "v",
}


def rgba(color, alpha):
    return (color[0], color[1], color[2], alpha)


def parse_system_name(dataset_name):
    name = str(dataset_name).lower()
    if name.startswith("wt"):
        return "wt"
    if name.startswith("sdu6"):
        return "sdu6"
    return name


def parse_replicate_id(dataset_name):
    name = str(dataset_name).lower()
    parts = name.split("-")
    if len(parts) >= 3 and parts[-1].isdigit():
        return int(parts[-1])
    return 1


def classify_residue(residue_name):
    name = str(residue_name).upper()
    return RESIDUE_CLASS.get(name, "polar")


def remove_outliers_iqr(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return values[(values >= lower) & (values <= upper)]


def significance_text(p_value):
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "ns"


def add_ribbon(ax, x0, x1, y0_top, y0_bottom, y1_top, y1_bottom, color, alpha=0.28):
    c1 = x0 + (x1 - x0) * 0.35
    c2 = x0 + (x1 - x0) * 0.65
    verts = [
        (x0, y0_top),
        (c1, y0_top),
        (c2, y1_top),
        (x1, y1_top),
        (x1, y1_bottom),
        (c2, y1_bottom),
        (c1, y0_bottom),
        (x0, y0_bottom),
        (x0, y0_top),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(patch)


def estimate_label_height(total_height, font_size):
    return total_height * (font_size / 72.0) * 0.038 * 6


def format_sx_residue_label(label):
    text = str(label)
    letters = "".join(ch for ch in text if ch.isalpha())
    digits = "".join(ch for ch in text if ch.isdigit())
    aa3 = letters[:3].capitalize()
    if aa3 and digits:
        return f"{aa3}{digits}"
    return text


def add_sx_residue_labels(ax, data, x_col, y_col):
    if data.empty:
        return

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_span = x_max - x_min
    y_span = y_max - y_min
    placed_boxes = []
    offsets = [
        (0.10, 0.04), (-0.10, 0.04), (0.10, -0.04), (-0.10, -0.04),
        (0.22, 0.08), (-0.22, 0.08), (0.22, -0.08), (-0.22, -0.08),
        (0.34, 0.12), (-0.34, 0.12), (0.34, -0.12), (-0.34, -0.12),
    ]

    for _, row in data.sort_values([y_col, x_col]).iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        label = format_sx_residue_label(row["label"])
        width = max(0.36, len(label) * x_span * 0.010)
        height = y_span * 0.035
        chosen = None

        for dx, dy in offsets:
            tx = min(max(x + dx, x_min + 0.03), x_max - width - 0.03)
            ty = min(max(y + dy, y_min + height / 2 + 0.02), y_max - height / 2 - 0.02)
            box = (tx, tx + width, ty - height / 2, ty + height / 2)
            overlaps = any(
                not (box[1] < other[0] or box[0] > other[1] or box[3] < other[2] or box[2] > other[3])
                for other in placed_boxes
            )
            if not overlaps:
                chosen = (tx, ty, box)
                break

        if chosen is None:
            tx = min(max(x + 0.10, x_min + 0.03), x_max - width - 0.03)
            ty = min(max(y + 0.04, y_min + height / 2 + 0.02), y_max - height / 2 - 0.02)
            chosen = (tx, ty, (tx, tx + width, ty - height / 2, ty + height / 2))

        tx, ty, box = chosen
        placed_boxes.append(box)
        use_arrow = abs(tx - x) > 0.12 or abs(ty - y) > 0.05
        ax.annotate(
            label,
            (x, y),
            xytext=(tx, ty),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=ANNOT_SIZE,
            arrowprops=(
                dict(arrowstyle="-", color="#555555", linewidth=0.35, shrinkA=0, shrinkB=2)
                if use_arrow else None
            ),
        )



# =========================
# Figure SX1: residue-type bar + sankey
# =========================

def collect_sankey_label_items(bottoms, totals, aa_order, min_block_height=2):
    """
    Collect label items for one side of the Sankey panel.

    Parameters
    ----------
    bottoms : dict
        Mapping from residue type to block bottom y-coordinate.
    totals : pd.Series or dict
        Mapping from residue type to block height.
    aa_order : list[str]
        Residue order from bottom to top.
    min_block_height : float
        Minimum block height required to draw a label.

    Returns
    -------
    list[dict]
        Each item contains:
        - aa
        - center_y
        - height
    """
    items = []
    for aa in aa_order:
        height = float(totals[aa])
        if height < min_block_height:
            continue
        center_y = float(bottoms[aa]) + height / 2
        items.append(
            {
                "aa": aa,
                "center_y": center_y,
                "height": height,
            }
        )
    return items


def solve_stacked_label_positions(label_items, min_gap, y_min, y_max):
    """
    Solve non-overlapping label positions for Sankey labels.

    Strategy
    --------
    1. Place labels from bottom to top.
    2. For each new label:
       y_new = max(block_center_y, previous_label_y + min_gap)
    3. If the top label exceeds y_max, shift all labels downward.
    4. Backward-pass to restore minimum spacing after shifting.
    5. If the bottom label falls below y_min, shift all labels upward.
    6. Forward-pass again to ensure spacing.

    Parameters
    ----------
    label_items : list[dict]
        Output of collect_sankey_label_items().
    min_gap : float
        Minimum gap between adjacent labels.
    y_min : float
        Lower boundary of label positions.
    y_max : float
        Upper boundary of label positions.

    Returns
    -------
    list[float]
        Solved y positions for text labels, in the same order as label_items.
    """
    if not label_items:
        return []

    placed_y = []

    # Pass 1: bottom -> top
    for i, item in enumerate(label_items):
        center_y = item["center_y"]
        if i == 0:
            y = max(center_y, y_min)
        else:
            y = max(center_y, placed_y[-1] + min_gap)
        placed_y.append(y)

    # Pass 2: if top exceeds boundary, shift all downward
    overflow_top = placed_y[-1] - y_max
    if overflow_top > 0:
        placed_y = [y - overflow_top for y in placed_y]

        # Backward pass to restore spacing
        for i in range(len(placed_y) - 2, -1, -1):
            placed_y[i] = min(placed_y[i], placed_y[i + 1] - min_gap)

    # Pass 3: if bottom falls below boundary, shift all upward
    overflow_bottom = y_min - placed_y[0]
    if overflow_bottom > 0:
        placed_y = [y + overflow_bottom for y in placed_y]

        # Forward pass to restore spacing
        for i in range(1, len(placed_y)):
            placed_y[i] = max(placed_y[i], placed_y[i - 1] + min_gap)

    return placed_y


def draw_sankey_side_labels(
    ax,
    label_items,
    label_y_positions,
    text_x,
    line_x_outer,
    line_x_inner,
    side,
    font_size,
    line_color="#4A4A4A",
    line_width=0.6,
):
    """
    Draw Sankey labels and leader lines for one side.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    label_items : list[dict]
    label_y_positions : list[float]
    text_x : float
        X position of text.
    line_x_outer : float
        X position near the text side.
    line_x_inner : float
        X position near the Sankey bar side.
    side : {"left", "right"}
        Text alignment / line direction.
    font_size : float
        Font size for residue labels.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    text_ha = "right" if side == "left" else "left"

    for item, y_text in zip(label_items, label_y_positions):
        aa = item["aa"]
        block_center_y = item["center_y"]

        ax.plot(
            [line_x_outer, line_x_inner],
            [y_text, block_center_y],
            color=line_color,
            linewidth=line_width,
            solid_capstyle="round",
        )
        ax.text(
            text_x,
            y_text,
            aa,
            ha=text_ha,
            va="center",
            fontsize=font_size,
        )


# Tunable parameters for Sankey labels
SANKEY_MIN_BLOCK_HEIGHT_FOR_LABEL = 1
SANKEY_LABEL_LINE_COLOR = "#4A4A4A"
SANKEY_LABEL_LINE_WIDTH = 0.6

SANKEY_LEFT_TEXT_OFFSET = 0.040
SANKEY_LEFT_LINE_OUTER_OFFSET = 0.035
SANKEY_LEFT_LINE_INNER_OFFSET = 0.005

SANKEY_RIGHT_TEXT_OFFSET = 0.040
SANKEY_RIGHT_LINE_INNER_OFFSET = 0.005
SANKEY_RIGHT_LINE_OUTER_OFFSET = 0.035


for mutant in ["sdu6"]:
    dist_file = INPUT_DIR / f"residue_type_summary_distribution_wt_{mutant}.csv"
    flow_file = INPUT_DIR / f"residue_type_summary_flow_wt_{mutant}.csv"

    dist_df = pd.read_csv(dist_file)
    flow_df = pd.read_csv(flow_file)

    dist_df["category"] = dist_df["category"].astype(str).str.upper()
    flow_df["reference_category"] = flow_df["reference_category"].astype(str).str.upper()
    flow_df["comparison_category"] = flow_df["comparison_category"].astype(str).str.upper()

    pivot_df = (
        dist_df.pivot_table(
            index="category",
            columns="series_dataset_id",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(AA_ORDER)
        .fillna(0)
    )

    wt_col = [c for c in pivot_df.columns if str(c).startswith("wt")][0]
    mutant_col = [c for c in pivot_df.columns if str(c).startswith(mutant)][0]

    flow_plot = (
        flow_df.groupby(["reference_category", "comparison_category"], as_index=False)["count"]
        .sum()
    )

    left_totals = (
        flow_plot.groupby("reference_category")["count"]
        .sum()
        .reindex(AA_ORDER_SANKEY, fill_value=0)
    )
    right_totals = (
        flow_plot.groupby("comparison_category")["count"]
        .sum()
        .reindex(AA_ORDER_SANKEY, fill_value=0)
    )
    total_count = max(left_totals.sum(), right_totals.sum())

    left_bottoms = {}
    current = 0.0
    for aa in AA_ORDER_SANKEY:
        left_bottoms[aa] = current
        current += float(left_totals[aa])

    right_bottoms = {}
    current = 0.0
    for aa in AA_ORDER_SANKEY:
        right_bottoms[aa] = current
        current += float(right_totals[aa])

    left_offsets = left_bottoms.copy()
    right_offsets = right_bottoms.copy()

    fig = plt.figure(figsize=FIG_BAR_SANKY)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.15)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_flow = fig.add_subplot(gs[0, 1])

    # -------------------------
    # Left panel: bar chart
    # -------------------------
    x = np.arange(len(AA_ORDER))
    width = 0.38
    ax_bar.bar(
        x - width / 2,
        pivot_df[wt_col].values,
        width=width,
        color=WT_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="wt",
    )
    ax_bar.bar(
        x + width / 2,
        pivot_df[mutant_col].values,
        width=width,
        color=SYSTEM_COLORS[mutant],
        edgecolor="black",
        linewidth=0.5,
        label=mutant,
    )
    ax_bar.set_xlabel("Residue type", fontsize=LABEL_SIZE)
    ax_bar.set_ylabel("Residue count", fontsize=LABEL_SIZE)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(AA_ORDER, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax_bar.tick_params(axis="y", labelsize=TICK_SIZE)
    ax_bar.set_xlim(-0.8, len(AA_ORDER) - 0.2)
    ax_bar.set_ylim(0, max(pivot_df[wt_col].max(), pivot_df[mutant_col].max()) * 1.15)
    ax_bar.legend(frameon=False, fontsize=LEGEND_SIZE, loc="upper right")

    for tick_label in ax_bar.get_xticklabels():
        if tick_label.get_text() in {"ALA", "ARG", "ASP", "GLU", "LEU", "LYS", "PRO"}:
            tick_label.set_fontweight("bold")

    # -------------------------
    # Right panel: Sankey bars
    # -------------------------
    bar_width = 0.15
    x_left0 = 0.10
    x_left1 = x_left0 + bar_width
    x_right1 = 0.90
    x_right0 = x_right1 - bar_width

    for aa in AA_ORDER_SANKEY:
        left_height = float(left_totals[aa])
        right_height = float(right_totals[aa])
        y0 = left_bottoms[aa]
        y1 = right_bottoms[aa]
        aa_class = classify_residue(aa)
        aa_color = CLASS_COLORS[aa_class]

        if left_height > 0:
            ax_flow.add_patch(
                Rectangle(
                    (x_left0, y0),
                    bar_width,
                    left_height,
                    facecolor=aa_color,
                    edgecolor="white",
                    linewidth=0.5,
                )
            )

        if right_height > 0:
            ax_flow.add_patch(
                Rectangle(
                    (x_right0, y1),
                    bar_width,
                    right_height,
                    facecolor=aa_color,
                    edgecolor="white",
                    linewidth=0.5,
                )
            )

    # -------------------------
    # Sankey labels: unified layout for left and right
    # -------------------------
    label_height = estimate_label_height(total_count, SANKEY_LABEL_FONT)
    min_label_gap = label_height * SANKEY_LABEL_GAP

    # Left labels
    left_label_items = collect_sankey_label_items(
        bottoms=left_bottoms,
        totals=left_totals,
        aa_order=AA_ORDER_SANKEY,
        min_block_height=SANKEY_MIN_BLOCK_HEIGHT_FOR_LABEL,
    )
    left_label_y = solve_stacked_label_positions(
        label_items=left_label_items,
        min_gap=min_label_gap,
        y_min=0,
        y_max=total_count,
    )
    draw_sankey_side_labels(
        ax=ax_flow,
        label_items=left_label_items,
        label_y_positions=left_label_y,
        text_x=x_left0 - SANKEY_LEFT_TEXT_OFFSET,
        line_x_outer=x_left0 - SANKEY_LEFT_LINE_OUTER_OFFSET,
        line_x_inner=x_left0 - SANKEY_LEFT_LINE_INNER_OFFSET,
        side="left",
        font_size=SANKEY_LABEL_FONT,
        line_color=SANKEY_LABEL_LINE_COLOR,
        line_width=SANKEY_LABEL_LINE_WIDTH,
    )

    # Right labels
    right_label_items = collect_sankey_label_items(
        bottoms=right_bottoms,
        totals=right_totals,
        aa_order=AA_ORDER_SANKEY,
        min_block_height=SANKEY_MIN_BLOCK_HEIGHT_FOR_LABEL,
    )
    right_label_y = solve_stacked_label_positions(
        label_items=right_label_items,
        min_gap=min_label_gap,
        y_min=0,
        y_max=total_count,
    )
    draw_sankey_side_labels(
        ax=ax_flow,
        label_items=right_label_items,
        label_y_positions=right_label_y,
        text_x=x_right1 + SANKEY_RIGHT_TEXT_OFFSET,
        line_x_outer=x_right1 + SANKEY_RIGHT_LINE_OUTER_OFFSET,
        line_x_inner=x_right1 + SANKEY_RIGHT_LINE_INNER_OFFSET,
        side="right",
        font_size=SANKEY_LABEL_FONT,
        line_color=SANKEY_LABEL_LINE_COLOR,
        line_width=SANKEY_LABEL_LINE_WIDTH,
    )

    # -------------------------
    # Sankey ribbons
    # -------------------------
    for _, row in flow_plot.iterrows():
        left_aa = row["reference_category"]
        right_aa = row["comparison_category"]
        value = float(row["count"])
        if value <= 0:
            continue

        left_top = left_offsets[left_aa] + value
        left_bottom = left_offsets[left_aa]
        right_top = right_offsets[right_aa] + value
        right_bottom = right_offsets[right_aa]

        aa_color = CLASS_COLORS[classify_residue(right_aa)]
        add_ribbon(
            ax_flow,
            x_left1,
            x_right0,
            left_top,
            left_bottom,
            right_top,
            right_bottom,
            aa_color,
            alpha=0.35,
        )

        left_offsets[left_aa] += value
        right_offsets[right_aa] += value

    # -------------------------
    # Axis styling
    # -------------------------
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, total_count)
    ax_flow.set_xlabel("Residue-type flow", fontsize=LABEL_SIZE)
    ax_flow.set_ylabel("Residue count", fontsize=LABEL_SIZE)
    ax_flow.set_xticks([x_left0 + bar_width / 2, x_right0 + bar_width / 2])
    ax_flow.set_xticklabels(["wt", mutant], fontsize=TICK_SIZE)
    ax_flow.tick_params(axis="y", labelsize=TICK_SIZE)
    ax_flow.spines["top"].set_visible(False)
    ax_flow.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.20, top=0.96)
    fig.savefig(
        OUTPUT_DIR / f"UGT84A56_SX1_residue_type_wt_vs_{mutant}.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

# =========================
# Figure 5B and SX2: delta KD vs delta SASA
# =========================
scatter_limits = {
    mutant: {
        "main_xlim": (-8.1, 8.1),
        "filtered_xlim": (-8.2, 8.2),
        "ylim": (-0.4, 0.4),
    }
    for mutant in ["sdu6"]
}
qp_compct_left, qp_compct_bottom = (0.82, 0.83)
quadrant_positions_compact = {
    "Q2": (qp_compct_left, qp_compct_bottom+0.06),
    "Q1": (qp_compct_left+0.11, qp_compct_bottom+0.06),
    "Q3": (qp_compct_left, qp_compct_bottom),
    "Q4": (qp_compct_left+0.11, qp_compct_bottom),
}

for mutant in ["sdu6"]:
    for suffix, stem, out_name in [
        ("", "Fig5B", f"UGT84A56_Fig5B_delta_kd_vs_delta_sasa_{mutant}.png"),
        ("-filtered", "SX2", f"UGT84A56_SX2_delta_kd_vs_delta_sasa_filtered_{mutant}.png"),
    ]:
        points_df = pd.read_csv(INPUT_DIR / f"surface_hydrophobicity_points_wt_{mutant}{suffix}.csv")
        stats_df = pd.read_csv(INPUT_DIR / f"surface_hydrophobicity_stats_wt_{mutant}{suffix}.csv")
        stats_row = stats_df.iloc[0]

        points_df["residue_class"] = points_df["experiment_residue_name"].apply(classify_residue)

        fig, ax = plt.subplots(figsize=FIG_SCATTER)
        xlim = scatter_limits[mutant]["filtered_xlim"] if suffix == "-filtered" else scatter_limits[mutant]["main_xlim"]
        ylim = scatter_limits[mutant]["ylim"]

        ax.add_patch(Rectangle((0, 0), xlim[1], ylim[1], facecolor="#F7F7F7", edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((xlim[0], 0), -xlim[0], ylim[1], facecolor="#FCFCFC", edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((xlim[0], ylim[0]), -xlim[0], -ylim[0], facecolor="#F7F7F7", edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((0, ylim[0]), xlim[1], -ylim[0], facecolor="#FCFCFC", edgecolor="none", zorder=0))
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
        reg_x = points_df["plot_x"].to_numpy(dtype=float)
        reg_y = points_df["plot_y"].to_numpy(dtype=float)
        reg_coef = np.polyfit(reg_x, reg_y, 1)
        reg_line_x = np.linspace(xlim[0], xlim[1], 300)
        reg_line_y = np.polyval(reg_coef, reg_line_x)
        ax.plot(reg_line_x, reg_line_y, color=SDU6_COLOR, linewidth=1.2, linestyle="--", zorder=1.5)

        legend_class_keys = []
        for residue_class in ["hydrophobic", "positive", "negative", "pro_special", "polar"]:
            sub = points_df.loc[points_df["residue_class"] == residue_class].copy()
            if sub.empty:
                continue
            legend_class_keys.append(residue_class)
            ax.scatter(
                sub["plot_x"],
                sub["plot_y"],
                s=36,
                marker=CLASS_MARKERS[residue_class],
                facecolor=CLASS_COLORS[residue_class],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.90,
                zorder=2,
            )

        stat_text = (
            r"Spearman’s $\rho$ = " + f"{stats_row['spearman_rho']:.3f}, " + \
            r"$P$" + f" = {stats_row['p_value']:.3f}\n"
            f"Quadrant concordance = {stats_row['consistency_ratio'] * 100:.1f}%"
        )
        ax.text(
            0.98,
            0.98,
            stat_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=ANNOT_SIZE,
            bbox=dict(boxstyle="square,pad=0.12", facecolor="none", edgecolor="none", alpha=0.70),
        )
        # quadrant_values = {
        #     "Q1": int(stats_row["q1"]),
        #     "Q2": int(stats_row["q2"]),
        #     "Q3": int(stats_row["q3"]),
        #     "Q4": int(stats_row["q4"]),
        # }
        # for q_name, (qx, qy) in quadrant_positions_compact.items():
        #     ax.text(
        #         qx,
        #         qy,
        #         f"{q_name}: {quadrant_values[q_name]}",
        #         transform=ax.transAxes,
        #         ha="center",
        #         va="top",
        #         fontsize=ANNOT_SIZE - 0.3,
        #     ) 

        ax.set_xlabel(r"$\Delta$ Kyte-Doolittle score", fontsize=LABEL_SIZE)
        ax.set_ylabel(r"$\Delta$ rSASA", fontsize=LABEL_SIZE)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(np.arange(-8.0, 8.1, 2.0))
        ax.set_yticks(np.arange(-0.4, 0.41, 0.1))
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        ax.set_box_aspect(1)

        class_label_map = {
            "hydrophobic": "Hydrophobic",
            "positive": "Positive",
            "negative": "Negative",
            "pro_special": "Pro" if (suffix == "-filtered" and "GLY" not in set(points_df["experiment_residue_name"].astype(str).str.upper())) else "Pro/Gly",
            "polar": "Polar neutral",
        }
        legend_handles = [
            Line2D(
                [0], [0],
                marker=CLASS_MARKERS[key],
                color="none",
                markerfacecolor=CLASS_COLORS[key],
                markeredgecolor="black",
                markersize=5,
                label=class_label_map[key],
            )
            for key in legend_class_keys
        ]
        ax.legend(handles=legend_handles, frameon=False, fontsize=6.5, ncol=2, loc="lower right")

        fig.subplots_adjust(left=0.15, right=0.98, bottom=0.16, top=0.97)
        fig.savefig(OUTPUT_DIR / out_name, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


rg_stats_rows = []
sasa_stats_rows = []


# =========================
# Figure SX3: RMSD time series
# =========================
rmsd_df = pd.read_csv(INPUT_DIR / "rmsd.csv")
rmsd_df["system_group"] = rmsd_df["dataset_name"].apply(parse_system_name)
rmsd_df["x_ns"] = rmsd_df["x_value"] / 1000.0

fig, ax = plt.subplots(figsize=FIG_WIDE)
legend_handles = []
for system in ["wt", "sdu6"]:
    sub = rmsd_df.loc[rmsd_df["system_group"] == system].copy()
    if sub.empty:
        continue
    for _, series in sub.groupby("dataset_name"):
        series = series.sort_values("x_value")
        ax.plot(
            series["x_ns"],
            series["y_value"],
            color=SYSTEM_COLORS[system],
            linewidth=1.2,
            alpha=0.65,
        )
    legend_handles.append(Line2D([0], [0], color=SYSTEM_COLORS[system], linewidth=2.0, label=system))

ax.set_xlabel("Time (ns)", fontsize=LABEL_SIZE)
ax.set_ylabel("RMSD (nm)", fontsize=LABEL_SIZE)
ax.tick_params(axis="both", labelsize=TICK_SIZE)
ax.set_xlim(rmsd_df["x_ns"].min(), rmsd_df["x_ns"].max())
ax.set_xticks(np.arange(0, rmsd_df["x_ns"].max() + 0.1, 10))
ax.set_ylim(0, rmsd_df["y_value"].max() * 1.08)
ax.legend(handles=legend_handles, frameon=False, fontsize=LEGEND_SIZE, ncol=4, loc="upper left")

fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.96)
fig.savefig(OUTPUT_DIR / "UGT84A56_SX3_RMSD_timecourse.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)


# =========================
# Figure SX8: RMSF residue profile (mean +/- SD)
# =========================
rmsf_df = pd.read_csv(INPUT_DIR / "RMSF.csv")
rmsf_df["system_group"] = rmsf_df["dataset_name"].apply(parse_system_name)
rmsf_df = rmsf_df.loc[rmsf_df["x_value"].between(1, 472)].copy()

fig, ax = plt.subplots(figsize=FIG_WIDE)
legend_handles = []
system_label_map = {
    "wt": "WT",
    "sdu6": "SDU6",
}

for system in ["wt", "sdu6"]:
    sub = rmsf_df.loc[rmsf_df["system_group"] == system].copy()
    if sub.empty:
        continue

    stats_df = (
        sub.groupby("x_value")["y_value"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("x_value")
    )
    x = stats_df["x_value"].to_numpy(dtype=float)
    mean = stats_df["mean"].to_numpy(dtype=float)
    std = stats_df["std"].fillna(0).to_numpy(dtype=float)
    color = SYSTEM_COLORS[system]

    ax.fill_between(
        x,
        np.clip(mean - std, 0, None),
        mean + std,
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(x, mean, color=color, linewidth=1.8)
    legend_handles.append(
        Line2D([0], [0], color=color, linewidth=2.2, label=system_label_map[system])
    )

ax.set_xlabel("Residue", fontsize=LABEL_SIZE)
ax.set_ylabel("RMSF (nm)", fontsize=LABEL_SIZE)
ax.tick_params(axis="both", labelsize=TICK_SIZE)
ax.set_xlim(1, 472)
ax.set_xticks([1, 100, 200, 300, 400, 472])
ax.set_ylim(0, 0.50)
ax.legend(handles=legend_handles, frameon=False, fontsize=LEGEND_SIZE, ncol=4, loc="upper left")

fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.96)
fig.savefig(OUTPUT_DIR / "UGT84A56_SX8_RMSF_mean_shadow.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)


# =========================
# Figure 5C: Delta G_bind vs distance
# =========================

distance_df = pd.read_csv(INPUT_DIR / "distance_vs_mmpbsa.csv")
distance_df["system_group"] = distance_df["dataset_name"].apply(parse_system_name)
distance_df = distance_df.loc[distance_df["y_value"] <= -0.01].copy()

FIG5C_SYSTEMS = ["wt", "sdu6"]
HBOND_DISTANCE = 0.35  # nm

fig5c_xy_stats_df = (
    distance_df.loc[distance_df["system_group"].isin(FIG5C_SYSTEMS)]
    .groupby("system_group", as_index=False)
    .agg(
        n_points=("x_value", "count"),
        x_value_mean=("x_value", "mean"),
        x_value_variance=("x_value", "var"),
        y_value_mean=("y_value", "mean"),
        y_value_variance=("y_value", "var"),
    )
)
fig5c_xy_stats_df.to_csv(
    OUTPUT_DIR / "UGT84A56_Fig5C_xy_mean_variance_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


def format_fig5c_axes(ax):
    """
    Apply unified axis formatting for Figure 5C.
    """
    ax.set_xlabel(r"$d$(His19-NE2, ESC-O6) (nm)", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\Delta G_{bind}$ (kcal/mol)", fontsize=LABEL_SIZE)
    ax.set_xlim(0.26, 2.0)
    ax.set_ylim(-30, 0.05)
    ax.set_xticks(np.arange(0.4, 2.01, 0.2))
    ax.set_yticks(np.arange(-30, 1, 5))
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    # Hydrogen-bond distance reference line
    ax.axvline(
        x=HBOND_DISTANCE,
        color="red",
        linestyle="--",
        linewidth=0.6,
        alpha=0.9,
        zorder=1,
    )


def make_legend_handles(system_list):
    """
    Create legend handles for the given systems.
    """
    handles = []
    for system in system_list:
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=SYSTEM_COLORS[system],
                markeredgecolor="none",
                markersize=6,
                alpha=0.75,
                label=system,
            )
        )
    return handles


def plot_single_system_scatter(system):
    """
    Plot a single-system scatter figure.
    """
    sub = distance_df.loc[distance_df["system_group"] == system].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=FIG_SCATTER)

    ax.scatter(
        sub["x_value"],
        sub["y_value"],
        s=12,
        color=SYSTEM_COLORS[system],
        edgecolors="none",
        alpha=0.45,
        zorder=2,
    )

    legend_handles = make_legend_handles([system])
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="upper right",
    )

    format_fig5c_axes(ax)

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.97)
    fig.savefig(
        OUTPUT_DIR / f"UGT84A56_Fig5C_deltaG_bind_vs_distance_{system}.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_overlay_scatter(system_list, output_name, alpha=0.35):
    """
    Plot an overlay scatter figure for multiple systems.

    Parameters
    ----------
    system_list : list[str]
        Systems to include in the overlay.
    output_name : str
        Output PNG filename.
    alpha : float
        Scatter transparency.
    """
    fig, ax = plt.subplots(figsize=FIG_SCATTER)
    plotted_systems = []

    for system in system_list:
        sub = distance_df.loc[distance_df["system_group"] == system].copy()
        if sub.empty:
            continue

        ax.scatter(
            sub["x_value"],
            sub["y_value"],
            s=12,
            color=SYSTEM_COLORS[system],
            edgecolors="none",
            alpha=alpha,
            zorder=2,
        )
        plotted_systems.append(system)

    if plotted_systems:
        legend_handles = make_legend_handles(plotted_systems)
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=LEGEND_SIZE,
            loc="upper right",
        )

    format_fig5c_axes(ax)

    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.97)
    fig.savefig(
        OUTPUT_DIR / output_name,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


# -------------------------
# 1) Single-system figures
# -------------------------
for system in FIG5C_SYSTEMS:
    plot_single_system_scatter(system)


# -------------------------
# 2) Overlay of all systems
# -------------------------
plot_overlay_scatter(
    system_list=FIG5C_SYSTEMS,
    output_name="UGT84A56_Fig5C_deltaG_bind_vs_distance_overlay.png",
    alpha=0.35,
)

# =========================
# Hydrogen-bond distance occupancy summary
# =========================
HBOND_DISTANCE = 0.35  # nm

hb_summary_rows = []
for system in ["wt", "sdu6"]:
    sub = distance_df.loc[distance_df["system_group"] == system].copy()
    if sub.empty:
        continue

    total_count = len(sub)
    hbond_count = int((sub["x_value"] < HBOND_DISTANCE).sum())
    hbond_fraction = hbond_count / total_count if total_count > 0 else np.nan
    hbond_percent = hbond_fraction * 100 if total_count > 0 else np.nan

    hb_summary_rows.append(
        {
            "system": system,
            "hbond_distance_cutoff_nm": HBOND_DISTANCE,
            "total_points": total_count,
            "points_below_cutoff": hbond_count,
            "fraction_below_cutoff": hbond_fraction,
            "percent_below_cutoff": hbond_percent,
        }
    )

hb_summary_df = pd.DataFrame(hb_summary_rows)
hb_summary_df.to_csv(
    OUTPUT_DIR / "UGT84A56_Fig5C_hbond_distance_fraction_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# Figure SX4: Rg violin
# frame-wise distributions for visualization
# significance tested on replicate-level means (unpaired Welch t-test)
# =========================
rg_df = pd.read_csv(INPUT_DIR / "Rg.csv")
rg_df["system_group"] = rg_df["dataset_name"].apply(parse_system_name)
rg_df["replicate_id"] = rg_df["dataset_name"].apply(parse_replicate_id)

rg_groups = []
rg_means = []
rg_stds = []
rg_positions = []
rg_ticklabels = []
rg_system_centers = []
rg_system_for_position = []
rg_system_pooled = {}
rg_replicate_means = {}
rg_replicate_stds = {}

position_cursor = 1.0
for system in ["wt", "sdu6"]:
    system_positions = []
    pooled_values = []
    rep_means = []
    rep_stds = []

    for rep in [1, 2, 3, 4]:
        sub = rg_df.loc[
            (rg_df["system_group"] == system) & (rg_df["replicate_id"] == rep),
            "y_value"
        ].values
        clean = np.asarray(sub, dtype=float)
        clean = clean[np.isfinite(clean)]
        clean = clean[clean <= 2.30]

        pooled_values.append(clean)
        rep_means.append(np.mean(clean))
        rep_stds.append(np.std(clean, ddof=1))

        rg_groups.append(clean)
        rg_means.append(np.mean(clean))
        rg_stds.append(np.std(clean, ddof=1))
        rg_positions.append(position_cursor)
        rg_ticklabels.append(str(rep))
        rg_system_for_position.append(system)
        system_positions.append(position_cursor)
        position_cursor += 1.0

    rg_system_centers.append(np.mean(system_positions))
    rg_system_pooled[system] = np.concatenate(pooled_values)
    rg_replicate_means[system] = np.asarray(rep_means, dtype=float)
    rg_replicate_stds[system] = np.asarray(rep_stds, dtype=float)

    position_cursor += 0.7

fig, ax = plt.subplots(figsize=FIG_VIOLIN_8)
parts = ax.violinplot(
    rg_groups,
    positions=rg_positions,
    widths=0.78,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

for body, system in zip(parts["bodies"], rg_system_for_position):
    body.set_facecolor(SYSTEM_COLORS[system])
    body.set_edgecolor("black")
    body.set_linewidth(0.8)
    body.set_alpha(0.75)

for i, pos in enumerate(rg_positions):
    ax.scatter(pos, rg_means[i], color="white", edgecolor="black", s=30, zorder=3)
    ax.vlines(
        pos,
        rg_means[i] - rg_stds[i],
        rg_means[i] + rg_stds[i],
        color="black",
        linewidth=1.1,
        zorder=3
    )
    ax.text(
        pos, 2.1525,
        f"{rg_means[i]:.3f}\n+/- {rg_stds[i]:.3f}",
        ha="center", va="bottom",
        fontsize=7.6, color="#333333", linespacing=0.95
    )

ax.set_ylabel("Radius of gyration (nm)", fontsize=LABEL_SIZE)
ax.set_xticks(rg_positions)
ax.set_xticklabels(rg_ticklabels, fontsize=9)
ax.tick_params(axis="y", labelsize=TICK_SIZE)
ax.set_xlim(min(rg_positions) - 0.8, max(rg_positions) + 1.8)
ax.set_ylim(2.15, 2.35)

for center, system in zip(rg_system_centers, ["wt", "sdu6"]):
    ax.text(center, -0.06, system, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=LABEL_SIZE)

legend_handles = [
    Patch(facecolor=SYSTEM_COLORS[system], edgecolor="black", alpha=0.75, label=system)
    for system in ["wt", "sdu6"]
]
legend_handles.append(
    Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
           linewidth=1.1, label="Mean +/- SD")
)
ax.legend(handles=legend_handles, frameon=False, fontsize=LEGEND_SIZE, ncol=3, loc="upper left")

fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.96)
fig.savefig(OUTPUT_DIR / "UGT84A56_SX4_Rg_violin_no_significance.png", dpi=DPI, bbox_inches="tight")

# -------------------------
# significance based on replicate-level means (unpaired Welch t-test)
# -------------------------
base_y = 2.300
step_y = 0.010
comparisons = [("sdu6", 1)]

for system, idx in comparisons:
    p_value = stats.ttest_ind(
        rg_replicate_means["wt"],
        rg_replicate_means[system],
        equal_var=False
    ).pvalue

    rg_stats_rows.append({
        "metric": "Rg",
        "group_a": "wt",
        "group_b": system,
        "n_a": len(rg_replicate_means["wt"]),
        "n_b": len(rg_replicate_means[system]),
        "mean_a": float(np.mean(rg_replicate_means["wt"])),
        "std_a": float(np.std(rg_replicate_means["wt"], ddof=1)),
        "mean_b": float(np.mean(rg_replicate_means[system])),
        "std_b": float(np.std(rg_replicate_means[system], ddof=1)),
        "test": "Welch t-test on replicate means",
        "p_value": float(p_value),
        "significance": significance_text(p_value),
        "filter_rule": "values <= 2.30 only; frame-wise values plotted; significance tested on replicate-level means",
    })

    x1 = rg_system_centers[0]
    x2 = rg_system_centers[idx]
    y = base_y + (idx - 1) * step_y
    ax.plot([x1, x1, x2, x2], [y - 0.0015, y, y, y - 0.0015], color="black", linewidth=0.9)
    ax.text((x1 + x2) / 2, y + 0.001, significance_text(p_value),
            ha="center", va="bottom", fontsize=9)

fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.96)
fig.savefig(OUTPUT_DIR / "UGT84A56_SX4_Rg_violin.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# =========================
# Figure SX5: total SASA violin
# frame-wise distributions for visualization
# significance tested on replicate-level means (unpaired Welch t-test)
# =========================
sasa_df = pd.read_csv(INPUT_DIR / "SASA.csv")
sasa_df["system_group"] = sasa_df["dataset_name"].apply(parse_system_name)
sasa_df["replicate_id"] = sasa_df["dataset_name"].apply(parse_replicate_id)

sasa_groups = []
sasa_means = []
sasa_stds = []
sasa_positions = []
sasa_ticklabels = []
sasa_system_centers = []
sasa_system_for_position = []
sasa_system_pooled = {}
sasa_replicate_means = {}
sasa_replicate_stds = {}

position_cursor = 1.0
for system in ["wt", "sdu6"]:
    system_positions = []
    pooled_values = []
    rep_means = []
    rep_stds = []

    for rep in [1, 2, 3, 4]:
        sub = sasa_df.loc[
            (sasa_df["system_group"] == system) & (sasa_df["replicate_id"] == rep),
            "y_value"
        ].values
        clean = np.asarray(sub, dtype=float)
        clean = clean[np.isfinite(clean)]
        clean = clean[(clean >= 190) & (clean <= 230)]

        pooled_values.append(clean)
        rep_means.append(np.mean(clean))
        rep_stds.append(np.std(clean, ddof=1))

        sasa_groups.append(clean)
        sasa_means.append(np.mean(clean))
        sasa_stds.append(np.std(clean, ddof=1))
        sasa_positions.append(position_cursor)
        sasa_ticklabels.append(str(rep))
        sasa_system_for_position.append(system)
        system_positions.append(position_cursor)
        position_cursor += 1.0

    sasa_system_centers.append(np.mean(system_positions))
    sasa_system_pooled[system] = np.concatenate(pooled_values)
    sasa_replicate_means[system] = np.asarray(rep_means, dtype=float)
    sasa_replicate_stds[system] = np.asarray(rep_stds, dtype=float)

    position_cursor += 0.7

fig, ax = plt.subplots(figsize=FIG_VIOLIN_8)
parts = ax.violinplot(
    sasa_groups,
    positions=sasa_positions,
    widths=0.78,
    showmeans=False,
    showmedians=False,
    showextrema=False
)

for body, system in zip(parts["bodies"], sasa_system_for_position):
    body.set_facecolor(SYSTEM_COLORS[system])
    body.set_edgecolor("black")
    body.set_linewidth(0.8)
    body.set_alpha(0.75)

for i, pos in enumerate(sasa_positions):
    ax.scatter(pos, sasa_means[i], color="white", edgecolor="black", s=30, zorder=3)
    ax.vlines(
        pos,
        sasa_means[i] - sasa_stds[i],
        sasa_means[i] + sasa_stds[i],
        color="black",
        linewidth=1.1,
        zorder=3
    )
    ax.text(
        pos, 191.0,
        f"{sasa_means[i]:.1f}\n+/- {sasa_stds[i]:.1f}",
        ha="center", va="bottom",
        fontsize=7.4, color="#333333", linespacing=0.95
    )

ax.set_ylabel("Total SASA (nm$^2$)", fontsize=LABEL_SIZE)
ax.set_xticks(sasa_positions)
ax.set_xticklabels(sasa_ticklabels, fontsize=9)
ax.tick_params(axis="y", labelsize=TICK_SIZE)
ax.set_xlim(min(sasa_positions) - 0.8, max(sasa_positions) + 1.8)
ax.set_ylim(190, 240)

for center, system in zip(sasa_system_centers, ["wt", "sdu6"]):
    ax.text(center, -0.06, system, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=LABEL_SIZE)

legend_handles = [
    Patch(facecolor=SYSTEM_COLORS[system], edgecolor="black", alpha=0.75, label=system)
    for system in ["wt", "sdu6"]
]
legend_handles.append(
    Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
           linewidth=1.1, label="Mean +/- SD")
)
ax.legend(handles=legend_handles, frameon=False, fontsize=LEGEND_SIZE, ncol=3, loc="upper left")

fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.96)
fig.savefig(OUTPUT_DIR / "UGT84A56_SX5_total_SASA_violin_no_significance.png", dpi=DPI, bbox_inches="tight")

# -------------------------
# significance based on replicate-level means (unpaired Welch t-test)
# -------------------------
base_y = 229.0
step_y = 2.4
comparisons = [("sdu6", 1)]

for system, idx in comparisons:
    p_value = stats.ttest_ind(
        sasa_replicate_means["wt"],
        sasa_replicate_means[system],
        equal_var=False
    ).pvalue

    sasa_stats_rows.append({
        "metric": "Total SASA",
        "group_a": "wt",
        "group_b": system,
        "n_a": len(sasa_replicate_means["wt"]),
        "n_b": len(sasa_replicate_means[system]),
        "mean_a": float(np.mean(sasa_replicate_means["wt"])),
        "std_a": float(np.std(sasa_replicate_means["wt"], ddof=1)),
        "mean_b": float(np.mean(sasa_replicate_means[system])),
        "std_b": float(np.std(sasa_replicate_means[system], ddof=1)),
        "test": "Welch t-test on replicate means",
        "p_value": float(p_value),
        "significance": significance_text(p_value),
        "filter_rule": "190 <= values <= 230; frame-wise values plotted; significance tested on replicate-level means",
    })

    x1 = sasa_system_centers[0]
    x2 = sasa_system_centers[idx]
    y = base_y + (idx - 1) * step_y
    ax.plot([x1, x1, x2, x2], [y - 0.5, y, y, y - 0.5], color="black", linewidth=0.9)
    ax.text((x1 + x2) / 2, y + 0.35, significance_text(p_value),
            ha="center", va="bottom", fontsize=9)

fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.96)
fig.savefig(OUTPUT_DIR / "UGT84A56_SX5_total_SASA_violin.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

pd.DataFrame(rg_stats_rows).to_csv(OUTPUT_DIR / "UGT84A56_SX4_Rg_stats.csv", index=False)
pd.DataFrame(sasa_stats_rows).to_csv(OUTPUT_DIR / "UGT84A56_SX5_total_SASA_stats.csv", index=False)

# =========================
# Figure SX6 + SX7: MMPBSA decomposition
# =========================
residues_df = pd.read_csv(INPUT_DIR / "residues.csv")
count_df = pd.read_csv(INPUT_DIR / "mmpbsa_count.csv")
avg_df = pd.read_csv(INPUT_DIR / "avg_mmpbsa_TOTAL.csv")
dataset_mmpbsa_df = pd.read_csv(INPUT_DIR / "mmpbsa_total_average_per_dataset.csv")
dataset_mmpbsa_df["system_group"] = dataset_mmpbsa_df["dataset_id"].apply(parse_system_name)

residues_df["label"] = residues_df["reference_residue_name"].astype(str).str.upper() + residues_df["aligned_residue_number"].astype(int).astype(str)
residues_df["label"] = residues_df["label"].str.replace("HISD", "HIS", regex=False)

count_plot = count_df.merge(
    residues_df[["project_residue_id", "aligned_residue_number", "reference_residue_name", "tags", "label"]],
    on=["project_residue_id", "aligned_residue_number"],
    how="left",
)
avg_plot = avg_df.merge(
    residues_df[["project_residue_id", "aligned_residue_number", "reference_residue_name", "tags", "label"]],
    on=["project_residue_id", "aligned_residue_number"],
    how="left",
)

system_residue_avg_df = (
    dataset_mmpbsa_df.dropna(subset=["value"])
    .groupby(["system_group", "project_residue_id", "aligned_residue_number"], as_index=False)
    .agg(avg_dataset_value=("value", "mean"))
)
residue_system_summary_df = (
    system_residue_avg_df
    .groupby(["project_residue_id", "aligned_residue_number"], as_index=False)
    .agg(
        avg_value=("avg_dataset_value", "mean"),
        avg_sd=("avg_dataset_value", "std"),
        count_value=("avg_dataset_value", "count"),
    )
)
avg_plot = residue_system_summary_df.merge(
    residues_df[["project_residue_id", "aligned_residue_number", "reference_residue_name", "tags", "label"]],
    on=["project_residue_id", "aligned_residue_number"],
    how="left",
)

# merged_positions = pd.concat(
#     [
#         count_plot[["project_residue_id", "aligned_residue_number", "label", "tags"]],
#         avg_plot[["project_residue_id", "aligned_residue_number", "label", "tags"]],
#     ],
#     ignore_index=True,
# ).drop_duplicates(subset=["project_residue_id"]).sort_values("aligned_residue_number")

merged_positions = pd.concat(
    [
        count_plot[["project_residue_id", "aligned_residue_number", "reference_residue_name", "label", "tags"]],
        avg_plot[["project_residue_id", "aligned_residue_number", "reference_residue_name", "label", "tags"]],
    ],
    ignore_index=True,
).drop_duplicates(subset=["project_residue_id"]).sort_values("aligned_residue_number")

count_plot = merged_positions.merge(
    avg_plot[["project_residue_id", "count_value"]],
    on="project_residue_id",
    how="left",
)

avg_plot = merged_positions.merge(
    avg_plot[["project_residue_id", "avg_value", "avg_sd"]],
    on="project_residue_id",
    how="left",
)

plot_df = count_plot.merge(avg_plot[["project_residue_id", "avg_value", "avg_sd"]], on="project_residue_id", how="left")
plot_df["count_value"] = plot_df["count_value"].fillna(0)
plot_df["has_dgbind"] = plot_df["avg_value"].notna()
plot_df["avg_value"] = plot_df["avg_value"].fillna(0)
plot_df["avg_sd"] = plot_df["avg_sd"].fillna(0)
plot_df["tags"] = plot_df["tags"].fillna("")
plot_df = plot_df.loc[~plot_df["label"].isin(["UDP477", "ESC478"])].copy()

plot_df["tags_lower"] = plot_df["tags"].astype(str).str.lower()
plot_df["is_conserved"] = plot_df["tags_lower"].str.contains("conserved")
plot_df["is_first_shell"] = plot_df["tags_lower"].str.contains("first-shell")
plot_df["is_second_shell"] = plot_df["tags_lower"].str.contains("second-shell")
plot_df["highlight"] = plot_df[["is_conserved", "is_first_shell", "is_second_shell"]].any(axis=1)

fig = plt.figure(figsize=FIG_MMPBSA)
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.55], hspace=0.10)
ax_top = fig.add_subplot(gs[0, 0])
ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax_annot = fig.add_subplot(gs[2, 0], sharex=ax_top)

x = np.arange(len(plot_df))
highlight_mask = plot_df["highlight"].values
count_colors = np.where(highlight_mask, "#5B5B5B", "#BEBEBE")
avg_colors = np.where(highlight_mask, "#1E9EFF", "#8DCDFE")

ax_top.bar(x, plot_df["count_value"], color=count_colors, edgecolor="black", linewidth=0.4)
ax_bottom.bar(
    x,
    plot_df["avg_value"],
    yerr=plot_df["avg_sd"],
    color=avg_colors,
    edgecolor="black",
    linewidth=0.4,
    error_kw={"ecolor": "black", "elinewidth": 0.6, "capsize": 2, "capthick": 0.6},
)

ax_top.set_ylabel("Count", fontsize=LABEL_SIZE)
ax_bottom.set_ylabel(r"$\Delta G_{bind}$ (kcal/mol)", fontsize=LABEL_SIZE)
ax_annot.set_xlabel("Residue", fontsize=LABEL_SIZE)

ax_top.tick_params(axis="y", labelsize=TICK_SIZE)
ax_bottom.tick_params(axis="y", labelsize=TICK_SIZE)
ax_annot.tick_params(axis="x", labelsize=8)
ax_top.set_xlim(-0.8, len(plot_df) - 0.2)
ax_top.set_ylim(0, 2.5)
ax_top.set_yticks([0, 1, 2])

avg_min = (plot_df["avg_value"] - plot_df["avg_sd"]).min()
avg_max = (plot_df["avg_value"] + plot_df["avg_sd"]).max()
avg_pad = max(0.05, (avg_max - avg_min) * 0.12)
ax_bottom.set_ylim(avg_min - avg_pad, avg_max + avg_pad)
ax_bottom.axhline(0, color="black", linewidth=0.8)

ax_annot.set_xticks(x)
ax_annot.set_xticklabels(plot_df["label"], rotation=90, ha="center")
ax_top.tick_params(axis="x", labelbottom=False)
ax_bottom.tick_params(axis="x", labelbottom=False)

for tick_label, is_highlight in zip(ax_annot.get_xticklabels(), plot_df["highlight"]):
    if is_highlight:
        tick_label.set_fontweight("bold")

ax_annot.set_ylim(-0.6, 2.6)
ax_annot.set_yticks([2, 1, 0])
ax_annot.set_yticklabels(["Conserved", "First shell", "Second shell"], fontsize=9)
ax_annot.tick_params(axis="y", length=0)
ax_annot.spines["top"].set_visible(False)
ax_annot.spines["right"].set_visible(False)

for row_y, col_name in [(2, "is_conserved"), (1, "is_first_shell"), (0, "is_second_shell")]:
    mask = plot_df[col_name].to_numpy(dtype=bool)
    ax_annot.scatter(x[~mask], np.full((~mask).sum(), row_y), s=36, facecolors="white", edgecolors="black", linewidths=0.8, zorder=2)
    ax_annot.scatter(x[mask], np.full(mask.sum(), row_y), s=36, facecolors="black", edgecolors="black", linewidths=0.8, zorder=3)

fig.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.97)
fig.savefig(OUTPUT_DIR / "UGT84A56_SX6_SX7_MMPBSA_decomposition.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# =========================
# Energy contribution summary for conserved / first shell / second shell
# =========================

# Per-residue total contribution
plot_df["total_energy"] = plot_df["avg_value"]

# Union mask: conserved OR first shell OR second shell
plot_df["is_selected_union"] = (
    plot_df["is_conserved"] |
    plot_df["is_first_shell"] |
    plot_df["is_second_shell"]
)

# Export per-residue details
residue_energy_export = plot_df[
    [
        "project_residue_id",
        "aligned_residue_number",
        "reference_residue_name",
        "label",
        "tags",
        "count_value",
        "avg_value",
        "avg_sd",
        "total_energy",
        "is_conserved",
        "is_first_shell",
        "is_second_shell",
        "is_selected_union",
    ]
].copy()

residue_energy_export.to_csv(
    OUTPUT_DIR / "UGT84A56_SX6_SX7_MMPBSA_residue_energy_details.csv",
    index=False,
    encoding="utf-8-sig",
)

# Summary statistics
selected_df = plot_df.loc[plot_df["is_selected_union"]].copy()
unselected_df = plot_df.loc[~plot_df["is_selected_union"]].copy()

total_energy_all = plot_df["total_energy"].sum()
total_energy_selected = selected_df["total_energy"].sum()
total_energy_unselected = unselected_df["total_energy"].sum()

total_abs_energy_all = plot_df["total_energy"].abs().sum()
total_abs_energy_selected = selected_df["total_energy"].abs().sum()
total_abs_energy_unselected = unselected_df["total_energy"].abs().sum()

summary_rows = [
    {
        "group": "selected_union",
        "definition": "conserved OR first shell OR second shell",
        "residue_count": int(len(selected_df)),
        "signed_total_energy": total_energy_selected,
        "signed_energy_ratio": (
            total_energy_selected / total_energy_all if total_energy_all != 0 else np.nan
        ),
        "absolute_total_energy": total_abs_energy_selected,
        "absolute_energy_ratio": (
            total_abs_energy_selected / total_abs_energy_all if total_abs_energy_all != 0 else np.nan
        ),
    },
    {
        "group": "other_residues",
        "definition": "NOT(conserved OR first shell OR second shell)",
        "residue_count": int(len(unselected_df)),
        "signed_total_energy": total_energy_unselected,
        "signed_energy_ratio": (
            total_energy_unselected / total_energy_all if total_energy_all != 0 else np.nan
        ),
        "absolute_total_energy": total_abs_energy_unselected,
        "absolute_energy_ratio": (
            total_abs_energy_unselected / total_abs_energy_all if total_abs_energy_all != 0 else np.nan
        ),
    },
    {
        "group": "all_residues",
        "definition": "all residues included in plot",
        "residue_count": int(len(plot_df)),
        "signed_total_energy": total_energy_all,
        "signed_energy_ratio": 1.0 if total_energy_all != 0 else np.nan,
        "absolute_total_energy": total_abs_energy_all,
        "absolute_energy_ratio": 1.0 if total_abs_energy_all != 0 else np.nan,
    },
]

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "UGT84A56_SX6_SX7_MMPBSA_energy_ratio_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# Figure SX9: substrate distance vs MMPBSA ΔGbind
# =========================

distance_df = pd.read_csv(INPUT_DIR / "residue_distances.csv")
sx9_df = plot_df.loc[plot_df["has_dgbind"]].merge(
    distance_df[
        [
            "residue_number",
            "any_atom_to_any_atom_min_angstrom",
        ]
    ],
    left_on="aligned_residue_number",
    right_on="residue_number",
    how="inner",
)

sx9_df["is_structure"] = sx9_df["is_first_shell"] | sx9_df["is_second_shell"]
sx9_df["selection_category"] = "None"
sx9_df.loc[sx9_df["is_conserved"] & ~sx9_df["is_structure"], "selection_category"] = "Sequence only"
sx9_df.loc[~sx9_df["is_conserved"] & sx9_df["is_structure"], "selection_category"] = "Structure only"
sx9_df.loc[sx9_df["is_conserved"] & sx9_df["is_structure"], "selection_category"] = "Structure + sequence"

sx9_df = sx9_df[
    [
        "project_residue_id",
        "aligned_residue_number",
        "label",
        "reference_residue_name",
        "tags",
        "count_value",
        "avg_value",
        "any_atom_to_any_atom_min_angstrom",
        "is_conserved",
        "is_first_shell",
        "is_second_shell",
        "selection_category",
    ]
].sort_values("aligned_residue_number")

sx9_df.to_csv(
    OUTPUT_DIR / "UGT84A56_SX9_distance_vs_dGbind_plot_data.csv",
    index=False,
    encoding="utf-8-sig",
)

def plot_sx9_distance_vs_dgbind(data, output_path, show_text=False):
    fig, ax = plt.subplots(figsize=FIG_SCATTER)

    ax.axhspan(-1.4, 0, xmin=0, xmax=1, color="#FCFCFC", zorder=0)
    ax.axhspan(0, 0.25, xmin=0, xmax=1, color="#F7F7F7", zorder=0)

    sx9_groups = [
        ("Fixed", data.loc[data["is_conserved"] | data["is_first_shell"] | data["is_second_shell"]], SDU6_COLOR, "s"),
        ("Variable", data.loc[~(data["is_conserved"] | data["is_first_shell"] | data["is_second_shell"])], "#8F8F8F", "o"),
    ]
    for label, group_df, color, marker in sx9_groups:
        if group_df.empty:
            continue
        ax.scatter(
            group_df["any_atom_to_any_atom_min_angstrom"],
            group_df["avg_value"],
            s=36,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.90,
            label=label,
            zorder=2,
        )

    if show_text:
        add_sx_residue_labels(ax, data, "any_atom_to_any_atom_min_angstrom", "avg_value")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(2, 12)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_ylim(-1.4, 0.25)
    ax.set_yticks([0.2, 0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4])
    ax.set_xlabel(r"$D_{ESC}$ (Å)", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\Delta G_{bind}$ (kcal/mol)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_box_aspect(1)

    legend_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=5,
            label=label,
        )
        for label, _, color, marker in sx9_groups
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=6.5, loc="lower right")
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.16, top=0.97)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


plot_sx9_distance_vs_dgbind(
    sx9_df,
    OUTPUT_DIR / "UGT84A56_SX9_distance_vs_dGbind_all.png",
    show_text=False,
)

plot_sx9_distance_vs_dgbind(
    sx9_df,
    OUTPUT_DIR / "UGT84A56_SX9_distance_vs_dGbind_all_with_text.png",
    show_text=True,
)

plot_sx9_distance_vs_dgbind(
    sx9_df.loc[sx9_df["count_value"] >= 2].copy(),
    OUTPUT_DIR / "UGT84A56_SX9_distance_vs_dGbind_count_gt_2.png",
    show_text=False,
)

plot_sx9_distance_vs_dgbind(
    sx9_df.loc[sx9_df["count_value"] >= 2].copy(),
    OUTPUT_DIR / "UGT84A56_SX9_distance_vs_dGbind_count_gt_2_with_text.png",
    show_text=True,
)

# =========================
# Figure SX10: per-system dataset ΔGbind vs substrate distance
# =========================

dataset_mmpbsa_df = pd.read_csv(INPUT_DIR / "mmpbsa_total_average_per_dataset.csv")
dataset_mmpbsa_df["system_group"] = dataset_mmpbsa_df["dataset_id"].apply(parse_system_name)

sx10_energy_df = (
    dataset_mmpbsa_df.dropna(subset=["value"])
    .groupby(["system_group", "project_residue_id", "aligned_residue_number"], as_index=False)
    .agg(
        avg_dataset_value=("value", "mean"),
        dataset_count=("value", "count"),
    )
)

sx10_base_df = sx10_energy_df.merge(
    plot_df[
        [
            "project_residue_id",
            "aligned_residue_number",
            "label",
            "reference_residue_name",
            "tags",
            "is_conserved",
            "is_first_shell",
            "is_second_shell",
        ]
    ],
    on=["project_residue_id", "aligned_residue_number"],
    how="left",
).merge(
    distance_df[
        [
            "residue_number",
            "any_atom_to_any_atom_min_angstrom",
        ]
    ],
    left_on="aligned_residue_number",
    right_on="residue_number",
    how="inner",
)

sx10_base_df["is_fixed"] = (
    sx10_base_df["is_conserved"] |
    sx10_base_df["is_first_shell"] |
    sx10_base_df["is_second_shell"]
)
sx10_base_df["selection_group"] = np.where(sx10_base_df["is_fixed"], "Fixed", "Variable")
sx10_base_df = sx10_base_df[
    [
        "system_group",
        "project_residue_id",
        "aligned_residue_number",
        "label",
        "reference_residue_name",
        "tags",
        "dataset_count",
        "avg_dataset_value",
        "any_atom_to_any_atom_min_angstrom",
        "is_conserved",
        "is_first_shell",
        "is_second_shell",
        "selection_group",
    ]
].sort_values(["system_group", "aligned_residue_number"])

sx10_base_df.to_csv(
    OUTPUT_DIR / "UGT84A56_SX10_distance_vs_dataset_dGbind_plot_data.csv",
    index=False,
    encoding="utf-8-sig",
)


def plot_sx10_distance_vs_dataset_dgbind(data, output_path, fixed_color, show_text=False):
    fig, ax = plt.subplots(figsize=FIG_SCATTER)
    plot_data = data.copy()
    low_outlier_mask = plot_data["avg_dataset_value"] < -1.4
    plot_data["plot_y"] = plot_data["avg_dataset_value"].where(~low_outlier_mask, -1.34)

    ax.axhspan(-1.4, 0, xmin=0, xmax=1, color="#FCFCFC", zorder=0)
    ax.axhspan(0, 0.25, xmin=0, xmax=1, color="#F7F7F7", zorder=0)

    sx10_groups = [
        ("Fixed", plot_data.loc[plot_data["selection_group"] == "Fixed"], fixed_color, "s"),
        ("Variable", plot_data.loc[plot_data["selection_group"] == "Variable"], "#8F8F8F", "o"),
    ]
    for label, group_df, color, marker in sx10_groups:
        if group_df.empty:
            continue
        ax.scatter(
            group_df["any_atom_to_any_atom_min_angstrom"],
            group_df["plot_y"],
            s=36,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.90,
            label=label,
            zorder=2,
        )

    if show_text:
        add_sx_residue_labels(ax, plot_data, "any_atom_to_any_atom_min_angstrom", "plot_y")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    if low_outlier_mask.any():
        for y0 in [0.050, 0.078]:
            ax.plot(
                [-0.018, 0.018],
                [y0 - 0.018, y0 + 0.018],
                transform=ax.transAxes,
                color="black",
                linewidth=0.8,
                clip_on=False,
            )
    ax.set_xlim(2, 12)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_ylim(-1.4, 0.25)
    ax.set_yticks([0.2, 0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4])
    ax.set_xlabel(r"$D_{ESC}$ (Å)", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\Delta G_{bind}$ (kcal/mol)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_box_aspect(1)

    legend_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=5,
            label=label,
        )
        for label, _, color, marker in sx10_groups
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=6.5, loc="lower right")
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.16, top=0.97)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


for system in ["wt", "sdu6"]:
    system_sx10_df = sx10_base_df.loc[sx10_base_df["system_group"] == system].copy()
    if system_sx10_df.empty:
        continue
    plot_sx10_distance_vs_dataset_dgbind(
        system_sx10_df,
        OUTPUT_DIR / f"UGT84A56_SX10_distance_vs_dataset_dGbind_{system}.png",
        SYSTEM_COLORS[system],
        show_text=False,
    )
    plot_sx10_distance_vs_dataset_dgbind(
        system_sx10_df,
        OUTPUT_DIR / f"UGT84A56_SX10_distance_vs_dataset_dGbind_{system}_with_text.png",
        SYSTEM_COLORS[system],
        show_text=True,
    )

print("Saved residue-level energy details to:",
      OUTPUT_DIR / "UGT84A56_SX6_SX7_MMPBSA_residue_energy_details.csv")
print("Saved summary ratios to:",
      OUTPUT_DIR / "UGT84A56_SX6_SX7_MMPBSA_energy_ratio_summary.csv")
