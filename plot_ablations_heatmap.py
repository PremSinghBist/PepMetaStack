#!/usr/bin/env python3
"""PepMetaStack ablation heatmaps for main-text and supplementary figures.
Outputs:
1. Main-text figure: leave-one-predictor-out (a), meta-regressors (b)
2. Supplementary figure: PLM, biological features, individual predictors, predictor families (a)-(d)
3. Additional horizontal figure: individual base predictors (a), predictor families and ensembles (b)
4. All six individual figures
Run from the project root:
    python plot_ablations_heatmap_main_supp.py
    python plot_ablations_heatmap_main_supp.py --dpi 600
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.patches import Rectangle
ROOT = Path.cwd()
DATA_DIR = ROOT / "results" / "ablation_results_data"
OUTPUT_DIR = ROOT / "results" / "plots" / "ablation_results_heatmap/new3"
INDIVIDUAL_SIZE = (4.3, 3.2166666667)  # original one-sixth panel-area scale
MAIN_TWO_PANEL_SIZE = (8.6, 3.85)
PREDICTOR_TWO_PANEL_SIZE = (8.6, 3.85)
SUPP_FOUR_PANEL_SIZE = (8.6, 6.9)
METRICS = {
    "rmse": (r"RMSE $\downarrow$", -1),
    "kendall_tau": (r"Kendall $\tau\uparrow$", 1),
    "r2": (r"$R^2\uparrow$", 1),
    "pearson_r": (r"Pearson $r\uparrow$", 1),
    "spearman_rho": (r"Spearman $\rho\uparrow$", 1),
    "mae": (r"MAE $\downarrow$", -1),
}
FOUR = ("rmse", "kendall_tau", "r2", "pearson_r")
SIX = ("pearson_r", "spearman_rho", "kendall_tau", "rmse", "mae", "r2")
@dataclass(frozen=True)
class Panel:
    tag: str
    name: str
    filename: str
    metrics: tuple[str, ...]
    reference: str
    labels: dict[str, str]
    filename_stub: str
PANELS = (
    Panel("a", "PLM representations and fusion", "01_plm_representations.csv", FOUR,
          "ProtT5", {
              "ESM-2 320-d": "ESM-2 (320-d)", "ESM-2 640-d": "ESM-2 (640-d)",
              "ESM-2 1280-d": "ESM-2 (1280-d)", "ProstT5": "ProstT5",
              "ProtT5": "ProtT5",
              "ProtT5+ESM, Concatenation": "ProtT5+ESM (concat.)",
              "ProtT5+ESM, Cross Attention": "ProtT5+ESM (cross-attn.)",
              "AMP-fine-tuned ProtT5": "AMP-tuned ProtT5",
          }, "01_plm_representations"),
    Panel("b", "Biological feature ablation", "02_biological_features.csv", FOUR,
          "PLM", {}, "02_biological_features"),
    Panel("c", "Individual base predictors", "03_base_predictors.csv", FOUR,
          "PepMetaStack (Full)", {
              "PepMetaStack (Full)": "PepMetaStack (full)",
              "HistGradientBoosting": "HistGB",
          }, "03_base_predictors"),
    Panel("d", "Leave-one-predictor-out", "04_leave_one_out.csv", FOUR,
          "PepMetaStack (Full)", {
              "PepMetaStack (Full)": "Full ensemble",
              "w/o XGBoost": "w/o XGBoost", "w/o Ridge": "w/o Ridge",
              "w/o MLP": "w/o MLP", "w/o KNN": "w/o KNN",
              "w/o HistGradientBoosting": "w/o HistGB",
              "w/o Extra Trees": "w/o Extra Trees", "w/o SVR": "w/o SVR",
          }, "04_leave_one_out"),
    Panel("e", "Predictor families and ensembles", "05_predictor_families.csv", FOUR,
          "PepMetaStack (Full)", {
              "PepMetaStack (Full)": "PepMetaStack (full)",
              "Heterogeneous-3 (SVR+ET+MLP)": "Hetero-3: SVR+ET+MLP",
              "Heterogeneous-2 (SVR+ET)": "Hetero-2: SVR+ET",
              "Distance-based (SVR+KNN)": "Distance: SVR+KNN",
              "Tree-based (ET+HGB+XGB)": "Trees: ET+HGB+XGB",
              "Heterogeneous-3 (Ridge+XGB+MLP)": "Hetero-3: Ridge+XGB+MLP",
          }, "05_predictor_families"),
    Panel("f", "Meta-regressor comparison", "06_meta_regressors.csv", SIX,
          "RidgeCV (Ours)", {"Linear Regression": "Linear regression",
                              "RidgeCV (Ours)": "RidgeCV (ours)"},
          "06_meta_regressors"),
)
COLORMAP = LinearSegmentedColormap.from_list(
    "pepmetastack_delta", ["#ad4949", "#efc6c0", "#f8f8f6", "#b9dadd", "#126b7a"])
NORM = SymLogNorm(linthresh=0.65, linscale=0.85, vmin=-15, vmax=15,
                  base=10, clip=True)
def configure_matplotlib():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.unicode_minus": False,
    })
def read_csv(panel: Panel, data_dir: Path) -> list[dict]:
    path = data_dir / panel.filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or sum(r.get("model") == panel.reference for r in rows) != 1:
        raise ValueError(f"Missing/nonunique reference {panel.reference!r} in {path}")
    for row in rows:
        for metric in panel.metrics:
            key, sdkey = f"{metric}_mean", f"{metric}_sd"
            if not row.get(key):
                raise ValueError(f"Missing {key} for {row['model']} in {path}")
            row[key] = float(row[key])
            row[sdkey] = float(row[sdkey]) if row.get(sdkey) else None
    return rows
def text_color(color):
    r, g, b = color[:3]
    return "#ffffff" if 0.2126*r + 0.7152*g + 0.0722*b < 0.54 else "#1b2b35"
def oriented_percent_change(value, reference, direction):
    if reference == 0:
        raise ZeroDivisionError("Percentage change requires a nonzero reference")
    return direction * (value - reference) / abs(reference) * 100
def draw_panel(fig, container, panel, rows, standalone=False, display_tag=None):
    x, y, w, h = container
    tag = panel.tag if display_tag is None else display_tag
    label_fraction = 0.36 if len(panel.metrics) == 4 else 0.30
    label_w = w * label_fraction
    fig.text(x, y + h - 0.012, f"({tag})  {panel.name}",
             ha="left", va="top", fontsize=7.5 if standalone else 8.2,
             weight="bold", color="#183847")
    top_band = h * (0.16 if standalone else 0.13)
    bottom_band = h * 0.015
    plot_y = y + bottom_band
    plot_h = h - top_band - bottom_band
    label_ax = fig.add_axes([x, plot_y, label_w, plot_h])
    cell_ax = fig.add_axes([x + label_w, plot_y, w - label_w, plot_h])
    n, m = len(rows), len(panel.metrics)
    for ax in (label_ax, cell_ax):
        ax.set_ylim(n, -0.90)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    label_ax.set_xlim(0, 1)
    cell_ax.set_xlim(0, m)
    label_font = 5.6 if standalone else 6.9
    value_font = (5.6 if standalone else 7.0) - (0.7 if m == 6 else 0)
    sd_font = value_font - 0.9
    hdr_font = (5.5 if standalone else 7.0) - (0.6 if m == 6 else 0)
    reference = next(r for r in rows if r["model"] == panel.reference)
    for i, row in enumerate(rows):
        is_reference = row["model"] == panel.reference
        row_bg = "#eaf0f2" if is_reference else ("#fafbfc" if i % 2 else "white")
        label_ax.add_patch(Rectangle((0, i), 1, 1, fc=row_bg, ec="none"))
        name = panel.labels.get(row["model"], row["model"])
        label_ax.text(.025, i + .5, name, ha="left", va="center",
                      fontsize=label_font, fontweight="bold" if is_reference else "normal",
                      color="#193847" if is_reference else "#34434e", clip_on=False)
        for j, metric in enumerate(panel.metrics):
            val, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
            delta = oriented_percent_change(val, reference[f"{metric}_mean"], METRICS[metric][1])
            face = COLORMAP(NORM(delta))
            cell_ax.add_patch(Rectangle((j, i), 1, 1, fc=face, ec="white", lw=.8))
            color = text_color(face)
            if sd is None:
                cell_ax.text(j + .5, i + .50, f"{val:.3f}", ha="center", va="center",
                             fontsize=value_font, color=color, fontweight="bold" if is_reference else "normal")
            else:
                cell_ax.text(j + .5, i + .34, f"{val:.3f}", ha="center", va="center",
                             fontsize=value_font, color=color, fontweight="bold" if is_reference else "normal")
                cell_ax.text(j + .5, i + .73, f"±{sd:.3f}", ha="center", va="center",
                             fontsize=sd_font, color=color)
        if is_reference:
            label_ax.add_patch(Rectangle((.004, i+.015), .996, .97, fill=False, ec="#516875", lw=.65))
            cell_ax.add_patch(Rectangle((0, i+.015), m-.012, .97, fill=False, ec="#516875", lw=.65))
    for j, metric in enumerate(panel.metrics):
        if m == 6:
            label = {"pearson_r": r"$r\uparrow$", "spearman_rho": r"$\rho\uparrow$",
                     "kendall_tau": r"$\tau\uparrow$", "rmse": r"RMSE$\downarrow$",
                     "mae": r"MAE$\downarrow$", "r2": r"$R^2\uparrow$"}[metric]
        else:
            label = {"rmse": r"RMSE$\downarrow$", "kendall_tau": r"$\tau\uparrow$",
                     "r2": r"$R^2\uparrow$", "pearson_r": r"$r\uparrow$"}[metric]
        cell_ax.text(j + .5, -.44, label, fontsize=hdr_font,
                     color="#243747", ha="center", va="center", fontweight="bold")
def add_colorbar(fig, area, standalone=False):
    x, y, w, h = area
    cax = fig.add_axes([x, y, w, h])
    mapper = mpl.cm.ScalarMappable(norm=NORM, cmap=COLORMAP)
    cb = fig.colorbar(mapper, cax=cax, orientation="horizontal", extend="both")
    ticks = [-15, -5, -1, 0, 1, 5, 15]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(t) if t < 0 else f"+{t}" if t > 0 else "0" for t in ticks])
    cb.ax.tick_params(labelsize=5.2 if standalone else 7, length=1.5, pad=1)
    cb.outline.set_linewidth(.4)
    cb.set_label("Change from reference (%)  ← worse | better →",
                 fontsize=5.5 if standalone else 7.4, labelpad=2)
def save_all(fig, out, stem, dpi):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        dest = out / f"{stem}.{ext}"
        fig.savefig(dest, dpi=dpi if ext == "png" else None,
                    facecolor="white", bbox_inches=None)
        print(f"Saved {dest}")
def make_main_two_panel(data, out, dpi):
    by_tag = {panel.tag: (panel, rows) for panel, rows in data}
    selected = [(replace(by_tag["d"][0], tag="a"), by_tag["d"][1]),
                (replace(by_tag["f"][0], tag="b"), by_tag["f"][1])]
    fig = plt.figure(figsize=MAIN_TWO_PANEL_SIZE, facecolor="white")
    left, right, top, bottom = .040, .990, .92, .205
    gap_x = .045
    panel_w = (right-left-gap_x)/2
    panel_h = top-bottom
    for i, (panel, rows) in enumerate(selected):
        x = left + i*(panel_w+gap_x)
        y = bottom
        draw_panel(fig, (x, y, panel_w, panel_h), panel, rows)
    add_colorbar(fig, (.24,.105,.52,.026))
    save_all(fig, out, "pepmetastack_main_leave_one_out_meta_2panel", dpi)
    plt.close(fig)
def make_predictor_two_panel(data, out, dpi):
    """Additional horizontal plot: individual base predictors and predictor families."""
    by_tag = {panel.tag: (panel, rows) for panel, rows in data}
    selected = [
        (replace(by_tag["c"][0], tag="a"), by_tag["c"][1]),
        (replace(by_tag["e"][0], tag="b"), by_tag["e"][1]),
    ]
    fig = plt.figure(figsize=PREDICTOR_TWO_PANEL_SIZE, facecolor="white")
    left, right, top, bottom = .040, .990, .92, .205
    gap_x = .045
    panel_w = (right - left - gap_x) / 2
    panel_h = top - bottom
    for i, (panel, rows) in enumerate(selected):
        x = left + i * (panel_w + gap_x)
        draw_panel(fig, (x, bottom, panel_w, panel_h), panel, rows)
    add_colorbar(fig, (.24, .105, .52, .026))
    save_all(fig, out, "pepmetastack_base_predictors_families_2panel", dpi)
    plt.close(fig)


def make_supp_four_panel(data, out, dpi):
    by_tag = {panel.tag: (panel, rows) for panel, rows in data}
    # Move predictor families into supplementary panel (d).
    selected = [(replace(by_tag[tag][0], tag=new_tag), by_tag[tag][1])
                for tag, new_tag in zip(("a", "b", "c", "e"), ("a", "b", "c", "d"))]
    fig = plt.figure(figsize=SUPP_FOUR_PANEL_SIZE, facecolor="white")
    left, right, top, bottom = .035, .988, .965, .145
    gap_x, gap_y = .034, .045
    panel_w = (right-left-gap_x)/2
    panel_h = (top-bottom-gap_y)/2
    for i, (panel, rows) in enumerate(selected):
        col, row = i % 2, i // 2
        x = left + col*(panel_w+gap_x)
        y = top - (row+1)*panel_h - row*gap_y
        draw_panel(fig, (x, y, panel_w, panel_h), panel, rows)
    add_colorbar(fig, (.21,.075,.58,.018))
    save_all(fig, out, "pepmetastack_supplementary_4panel", dpi)
    plt.close(fig)
def make_standalone(data, out, dpi):
    for panel, rows in data:
        fig = plt.figure(figsize=INDIVIDUAL_SIZE, facecolor="white")
        draw_panel(fig, (.035,.20,.93,.76), panel, rows, standalone=True)
        add_colorbar(fig, (.20,.103,.60,.015), standalone=True)
        fig.text(.5,.022,"Outlined row: reference; colors show relative change.",
                 ha="center", fontsize=5.2, color="#52636b")
        save_all(fig,out,panel.filename_stub,dpi)
        plt.close(fig)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    configure_matplotlib()
    data = [(panel, read_csv(panel, args.data_dir)) for panel in PANELS]
    make_main_two_panel(data, args.output_dir, args.dpi)
    make_predictor_two_panel(data, args.output_dir, args.dpi)
    make_supp_four_panel(data, args.output_dir, args.dpi)
    make_standalone(data, args.output_dir / "individual", args.dpi)
    print("Done: main 2-panel + predictor 2-panel + supplementary 4-panel + all 6 individual figures")
if __name__ == "__main__":
    '''
    python plot_ablations_heatmap.py --dpi 600
    '''
    main()
