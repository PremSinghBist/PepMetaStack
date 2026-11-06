#!/usr/bin/env python3
"""

Uses PepMetaStack's saved FINAL independent-test predictions; does not retrain.

The pipeline's stacking_final_independent_predictions.csv contains only
    y_true, y_pred
so peptide sequences must be recovered from the same test split, IN THE
SAME ROW ORDER, by supplying --dataset-csv (via data_util.get_cv_split,
fold=1), or --test-csv (exported independent test subset). Alternatively,
the prediction CSV may already contain a 'sequence' column.

Examples
--------
# Run alongside data_util.py (or add --project-dir PATH):
python plot_activity_length_error.py \
  --predictions ./results/RIDGE_XG_BOOST_EXCLUDED/stacking_final_independent_predictions.csv \
  --dataset-csv ./data/ecoli_five_fold_dataset.csv \
  --output-dir ./results/plots/activity_length_error \
  --activity-label 'log10 MIC' --dpi 600

# If a separate independent test CSV is available:
python plot_activity_length_error.py --predictions predictions.csv \
  --test-csv independent_test.csv --output-dir ./results/plots/activity_length_error

Creates a horizontal, one-row two-column figure:
(a) MAE by EXPERIMENTAL activity quantile.
(b) MAE by peptide-length quantile (amino-acid count).
Dots are mean absolute errors; error bars are within-bin percentile-bootstrap
95% intervals. Dashed line = overall MAE. Bin sample counts appear above dots.
Raw per-peptide and per-bin analysis data are saved to CSV. Optional --scatter
adds a light scatter cloud of per-peptide absolute errors.

Notes
-----
* Quantile bins are based ONLY on observed activity / peptide length.
* Repeated integer lengths may reduce the number of unique bins. Identical
  lengths are never artificially split between bins.
* Bootstrap intervals describe uncertainty conditional on the observed
  test data, not variability from retraining, dataset shifts, or model search.
* For MIC expressed on a log10 scale, error and MAE are also in log10 units.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


BLUE = '#20668D'
GREY = '#666666'


def _checked_values(frame: pd.DataFrame, column: str, source: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f'{source} lacks column {column!r}. Available: {list(frame)}')
    try:
        out = pd.to_numeric(frame[column], errors='raise').to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{source}: {column!r} must be numeric') from exc
    if not np.isfinite(out).all():
        raise ValueError(f'{source}: {column!r} contains nonfinite values')
    return out


def _checked_sequences(frame: pd.DataFrame, column: str, source: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f'{source} lacks sequence column {column!r}')
    seq = frame[column]
    if seq.isna().any() or any(not isinstance(s, str) or not s.strip() for s in seq):
        raise ValueError(f'{source}: sequences must be nonempty strings')
    return seq.astype(str).to_numpy()


def load_predictions_with_sequences(
    prediction_csv: Path,
    *,
    test_csv: Path | None,
    dataset_csv: Path | None,
    project_dir: Path | None,
    sequence_col: str,
    target_col: str,
    test_fold: int,
) -> pd.DataFrame:
    """Use saved predictions and verify corresponding test rows, when supplied."""
    if not prediction_csv.is_file():
        raise FileNotFoundError(f'Prediction CSV not found: {prediction_csv}')
    prediction = pd.read_csv(prediction_csv)
    if len(prediction) < 2:
        raise ValueError('At least two predictions are required')
    truth = _checked_values(prediction, 'y_true', str(prediction_csv))
    predicted = _checked_values(prediction, 'y_pred', str(prediction_csv))

    if test_csv is not None and dataset_csv is not None:
        raise ValueError('Choose --test-csv OR --dataset-csv, not both')

    test = None
    if test_csv is not None:
        if not test_csv.is_file():
            raise FileNotFoundError(f'Independent test CSV not found: {test_csv}')
        test = pd.read_csv(test_csv)
    elif dataset_csv is not None:
        if not dataset_csv.is_file():
            raise FileNotFoundError(f'Dataset CSV not found: {dataset_csv}')
        if project_dir is not None:
            sys.path.insert(0, str(project_dir.expanduser().resolve()))
        try:
            data_util = importlib.import_module('data_util')
        except ModuleNotFoundError as exc:
            raise ImportError('Cannot import data_util.py. Run in the project folder '
                              'or supply --project-dir PATH, or use --test-csv.') from exc
        if not hasattr(data_util, 'get_cv_split'):
            raise AttributeError('data_util.py must define get_cv_split')
        _, _, test = data_util.get_cv_split(input_csv_path=str(dataset_csv), fold=test_fold)
        test = test.reset_index(drop=True)

    if test is None:
        if sequence_col not in prediction:
            raise ValueError(
                'Prediction CSV has no sequences. Provide --dataset-csv '
                '(with --project-dir if necessary) or --test-csv to recover '
                'peptide lengths. The pipeline saves only y_true/y_pred.'
            )
        sequences = _checked_sequences(prediction, sequence_col, 'prediction CSV')
    else:
        if len(test) != len(prediction):
            raise ValueError(f'Test/prediction length mismatch: {len(test)} vs {len(prediction)}')
        test_truth = _checked_values(test, target_col, 'independent test split')
        if not np.allclose(truth, test_truth, atol=1e-5, rtol=1e-5):
            mismatches = int(np.sum(~np.isclose(truth, test_truth, atol=1e-5, rtol=1e-5)))
            raise ValueError(
                f'{mismatches} experimental values do not match the same row '
                'in the test split. The predictions and sequences MUST come '
                'from identical ordered test rows; do not join on MIC values.'
            )
        sequences = _checked_sequences(test, sequence_col, 'independent test split')
        if sequence_col in prediction:
            original = _checked_sequences(prediction, sequence_col, 'prediction CSV')
            if not np.array_equal(original, sequences):
                raise ValueError('Sequence identifiers differ between the two CSVs')

    lengths = np.array([len(seq.strip()) for seq in sequences], dtype=int)
    if np.any(lengths < 1):
        raise ValueError('Sequences must contain at least one amino acid')
    return pd.DataFrame({
        'sequence': sequences,
        'peptide_length_aa': lengths,
        'y_true': truth,
        'y_pred': predicted,
        'signed_residual': truth - predicted,
        'absolute_error': np.abs(truth - predicted),
    })


def _bootstrap_mae_ci(values: np.ndarray, *, rng: np.random.Generator,
                      iterations: int) -> tuple[float, float]:
    """Percentile bootstrap for conditional mean absolute error, bounded memory."""
    if len(values) < 2 or iterations <= 0:
        return float('nan'), float('nan')
    statistics = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 128):
        amount = min(128, iterations - start)
        indices = rng.integers(0, len(values), size=(amount, len(values)))
        statistics[start:start+amount] = np.mean(values[indices], axis=1)
    low, high = np.quantile(statistics, [0.025, 0.975])
    return float(low), float(high)


def binned_error_summary(data: pd.DataFrame, column: str, *,
                         bins: int, bootstraps: int, seed: int) -> pd.DataFrame:
    x = data[column].to_numpy(dtype=float)
    e = data['absolute_error'].to_numpy(dtype=float)
    signed = data['signed_residual'].to_numpy(dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        raise ValueError(f'{column} is constant; cannot analyze variation')
    if len(edges) - 1 < bins:
        print(f'Note: {column} has {len(edges)-1} nonempty-capable quantile '
              f'intervals rather than {bins}, due to repeated values.')

    # Digitize at internal boundaries only, including both extremes.
    groups = np.digitize(x, edges[1:-1], right=True)
    random = np.random.default_rng(seed)
    summary = []
    for i in range(len(edges) - 1):
        idx = groups == i
        n = int(idx.sum())
        if n == 0:
            continue
        xe = x[idx]
        residual = e[idx]
        lo_ci, hi_ci = _bootstrap_mae_ci(residual, rng=random, iterations=bootstraps)
        summary.append({
            'bin': i + 1,
            'variable': column,
            'interval_min': float(xe.min()),
            'interval_max': float(xe.max()),
            'bin_edge_low': float(edges[i]),
            'bin_edge_high': float(edges[i+1]),
            'x_median': float(np.median(xe)),
            'n': n,
            'mae': float(np.mean(residual)),
            'rmse': float(np.sqrt(np.mean(residual**2))),
            'median_absolute_error': float(np.median(residual)),
            'mean_signed_residual': float(np.mean(signed[idx])),
            'mae_ci_low': lo_ci,
            'mae_ci_high': hi_ci,
        })
    result = pd.DataFrame(summary)
    if len(result) < 2:
        raise ValueError(f'{column} yields fewer than two populated bins')
    assert result['n'].sum() == len(data), 'Some peptides were omitted from binning'
    return result


def _draw_panel(ax, *, data: pd.DataFrame, summary: pd.DataFrame,
                x_col: str, xlabel: str, panel: str, overall_mae: float,
                scatter: bool, activity_label: str, show_ci: bool):
    x = summary['x_median'].to_numpy(float)
    mae = summary['mae'].to_numpy(float)
    lo = summary['mae_ci_low'].to_numpy(float)
    hi = summary['mae_ci_high'].to_numpy(float)

    if scatter:
        raw_x = data[x_col].to_numpy(float)
        raw_y = data['absolute_error'].to_numpy(float)
        ax.scatter(raw_x, raw_y, s=3.2, alpha=0.09, color=BLUE,
                   edgecolors='none', rasterized=True, zorder=1)

    ax.axhline(overall_mae, color='0.45', linestyle='--',
               linewidth=0.8, zorder=2)
    ok = np.isfinite(lo) & np.isfinite(hi)
    if np.any(ok):
        yerr = np.vstack([mae[ok] - lo[ok], hi[ok] - mae[ok]])
        ax.errorbar(x[ok], mae[ok], yerr=yerr, color=BLUE,
                    fmt='none', ecolor=BLUE, capsize=2.2,
                    elinewidth=0.9, capthick=0.85, zorder=4)
    ax.plot(x, mae, '-o', color=BLUE, linewidth=1.3, markersize=4.0,
            markeredgecolor='white', markeredgewidth=0.45, zorder=5)
    # Keep every binned point distinct even with integer length values.
    # The summary points are located at bin medians. Frame those medians
    # tightly when displaying only the binned summary; do not imply that
    # the most extreme observations were omitted from the bin analysis.
    xmin = float(data[x_col].min()) if scatter else float(np.min(x))
    xmax = float(data[x_col].max()) if scatter else float(np.max(x))
    if xmax > xmin:
        ax.set_xlim(xmin - .075*(xmax-xmin), xmax + .075*(xmax-xmin))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=(x_col == 'peptide_length_aa')))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel(f'MAE ({activity_label})', labelpad=4)

    # Annotation is placed above the panel rather than on top of data.
    ax.text(0.015, 0.985, panel, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left',
            bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.9, 'pad':1},
            zorder=8)
    counts = summary['n'].to_numpy(int)
    count_note = (f'n/bin: {counts.min()}–{counts.max()}'
                  if counts.min() != counts.max() else f'n/bin: {counts.min()}')
    ax.text(0.985, 0.985,
            f'Overall MAE = {overall_mae:.3f}\n{count_note}',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.4,
            color=GREY, linespacing=1.5,
            bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':1},
            zorder=8)

    if not scatter:
        bound_high = max(float(np.max(mae)), float(np.nanmax(hi)) if ok.any() else 0., overall_mae)
        ax.set_ylim(0, bound_high * 1.22 if bound_high > 0 else .1)
    else:
        ax.set_ylim(bottom=0)
        ax.text(.985, .04, 'Dots: peptides; line: binned MAE' + (' (95% CI)' if show_ci else ''),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5,
                color=GREY, bbox={'facecolor':'white','edgecolor':'none', 'alpha':.88,'pad':1})

    ax.grid(axis='y', color='0.87', linestyle=':', linewidth=.45, zorder=0)
    ax.tick_params(direction='out', length=2.5, width=.65)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_activity_length_error(prediction_csv: str | Path,
                               output_dir: str | Path,
                               *,
                               test_csv: str | Path | None = None,
                               dataset_csv: str | Path | None = None,
                               project_dir: str | Path | None = None,
                               sequence_col: str = 'sequence',
                               target_col: str = 'value',
                               test_fold: int = 1,
                               filename: str = 'figure_activity_length_error',
                               activity_label: str = 'log10 MIC',
                               n_bins: int = 5,
                               n_bootstraps: int = 1000,
                               seed: int = 42,
                               scatter: bool = False,
                               width_in: float = 3.35,
                               height_in: float = 5.0,
                               dpi: int = 600) -> dict[str, Path]:
    if width_in <= 0 or height_in <= 0 or n_bins < 2 or n_bootstraps < 0:
        raise ValueError('Check figure dimensions, bins (>=2), bootstrap iterations (>=0)')
    if dpi <= 300:
        raise ValueError('DPI must exceed 300 (600 recommended)')

    data = load_predictions_with_sequences(
        Path(prediction_csv), test_csv=Path(test_csv) if test_csv else None,
        dataset_csv=Path(dataset_csv) if dataset_csv else None,
        project_dir=Path(project_dir) if project_dir else None,
        sequence_col=sequence_col, target_col=target_col, test_fold=test_fold)
    activity_summary = binned_error_summary(data, 'y_true',
        bins=n_bins, bootstraps=n_bootstraps, seed=seed)
    length_summary = binned_error_summary(data, 'peptide_length_aa',
        bins=n_bins, bootstraps=n_bootstraps, seed=seed + 1)
    overall_mae = float(data['absolute_error'].mean())
    overall_rmse = float(np.sqrt(np.mean(data['signed_residual'].to_numpy(float)**2)))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    settings = {
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': .65, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': dpi, 'svg.fonttype': 'none',
    }
    with plt.rc_context(settings):
        # Horizontal 1-row x 2-column layout for journal presentation.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_in, height_in),
                                     gridspec_kw={'width_ratios': [1, 1]})
        _draw_panel(ax1, data=data, summary=activity_summary, x_col='y_true',
                    xlabel=f'Experimental {activity_label}', panel='(a)',
                    overall_mae=overall_mae, scatter=scatter,
                    activity_label=activity_label, show_ci=n_bootstraps > 0)
        _draw_panel(ax2, data=data, summary=length_summary, x_col='peptide_length_aa',
                    xlabel='Peptide length (aa)', panel='(b)',
                    overall_mae=overall_mae, scatter=scatter,
                    activity_label=activity_label, show_ci=n_bootstraps > 0)
        # Keep the two plot areas identical and reserve room for y-axis labels.
        # Both axes use the same bottom/top margins; wspace controls only the gap.
        fig.subplots_adjust(left=.095, right=.985, top=.95, bottom=.20, wspace=.36)
        saved = {}
        for ext in ('png', 'pdf', 'svg'):
            path = out / f'{filename}.{ext}'
            fig.savefig(path, dpi=dpi, facecolor='white')
            saved[ext] = path
        plt.close(fig)

    data.to_csv(out / f'{filename}_per_peptide.csv', index=False)
    activity_summary.to_csv(out / f'{filename}_activity_bins.csv', index=False)
    length_summary.to_csv(out / f'{filename}_length_bins.csv', index=False)
    print(f'Analyzed {len(data)} independent-test peptides')
    print(f'Overall MAE={overall_mae:.4f}  RMSE={overall_rmse:.4f}')
    print('\nActivity-dependent errors:\n', activity_summary[
        ['interval_min', 'interval_max', 'n', 'mae', 'mae_ci_low', 'mae_ci_high']].to_string(index=False))
    print('\nLength-dependent errors:\n', length_summary[
        ['interval_min', 'interval_max', 'n', 'mae', 'mae_ci_low', 'mae_ci_high']].to_string(index=False))
    for format_name, filepath in saved.items():
        print(f'Saved {format_name.upper()}: {filepath}')
    print(f'Saved per-peptide and binned analysis CSVs in {out}')
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', required=True, type=Path,
                        help='CSV with y_true and y_pred; optional sequence')
    parser.add_argument('--test-csv', type=Path,
                        help='Ordered independent-test CSV with sequence and value')
    parser.add_argument('--dataset-csv', type=Path,
                        help='Original dataset read by data_util.get_cv_split (fold=1)')
    parser.add_argument('--project-dir', type=Path,
                        help='Directory containing your data_util.py module')
    parser.add_argument('--test-fold', type=int, default=1)
    parser.add_argument('--sequence-col', default='sequence')
    parser.add_argument('--target-col', default='value')
    parser.add_argument('--output-dir', type=Path, default=Path('./results/plots'))
    parser.add_argument('--filename', default='figure_activity_length_error')
    parser.add_argument('--activity-label', default='log10 MIC')
    parser.add_argument('--bins', type=int, default=5)
    parser.add_argument('--bootstraps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scatter', action='store_true',
                        help='Add faint per-peptide absolute error points')
    parser.add_argument('--width', type=float, default=7.2)
    parser.add_argument('--height', type=float, default=2.4)
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()
    plot_activity_length_error(
        args.predictions, args.output_dir,
        test_csv=args.test_csv, dataset_csv=args.dataset_csv,
        project_dir=args.project_dir, sequence_col=args.sequence_col,
        target_col=args.target_col, test_fold=args.test_fold,
        filename=args.filename, activity_label=args.activity_label,
        n_bins=args.bins, n_bootstraps=args.bootstraps, seed=args.seed,
        scatter=args.scatter, width_in=args.width,
        height_in=args.height, dpi=args.dpi)


if __name__ == '__main__':
    '''
    python plot_activity_length_error_horizontal.py \
  --predictions ./results/RIDGE_XG_BOOST_EXCLUDED/stacking_final_independent_predictions.csv \
  --dataset-csv ./data/ecoli_five_fold_dataset.csv \
  --output-dir ./results/plots/activity_length_error_horizontal_1 \
  --dpi 600
    '''
    
    main()
