#!/usr/bin/env python3
"""Publication-quality comparison of ProtT5 and ESM2 across microbial species.

Input CSV columns:
  species, embedding, and <metric>_mean, <metric>_std for each metric:
  Pearson_R, Spearman_R, Kendall_Tau, RMSE, MAE, R2

Usage:
  python plot_species_performance.py --input validation_mean_sd.csv \
      --output results/species_comparison

Outputs:
  species_performance_comparison.pdf/.svg/.png

Requires: pandas, numpy, matplotlib
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

METRICS = [
    ('Pearson_R', "Pearson's $r$", True),
    ('Spearman_R', "Spearman's $\\rho$", True),
    ('Kendall_Tau', "Kendall's $\\tau$", True),
    ('RMSE', 'RMSE', False),
    ('MAE', 'MAE', False),
    ('R2', r'$\mathbf{R}^{\mathbf{2}}$', True), #Bold for math mode
    # ('R2', '$R^2$', True),
]
PALETTE = {'ProtT5': '#2474A6', 'ESM2': '#D66B43'}
SPECIES = {
    'Staphylococcus aureus': r'$S.\ aureus$',
    'Pseudomonas aeruginosa': r'$P.\ aeruginosa$',
    'C albicans': r'$C.\ albicans$',
    'Candida albicans': r'$C.\ albicans$',
    'Bacillus subtilis': r'$B.\ subtilis$',
    'Staphylococcus epidermidis': r'$S.\ epidermidis$',
}


def make_figure(df: pd.DataFrame, output: Path, dpi: int = 600,
                species_order: list[str] | None = None) -> None:
    required = {'species', 'embedding'} | {
        f'{m}_{s}' for m, _, _ in METRICS for s in ('mean', 'std')
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError('Missing required CSV columns: ' + ', '.join(missing))
    df = df.copy()
    df['embedding'] = df['embedding'].astype(str).str.strip().str.lower().map({
        'prott5': 'ProtT5', 'prot_t5': 'ProtT5',
        'esm2': 'ESM2', 'esm-2': 'ESM2',
    })
    if df['embedding'].isna().any():
        raise ValueError('Only embeddings "prott5" and "esm2" are expected.')
    df['species'] = df['species'].astype(str).str.strip()
    for metric, _, _ in METRICS:
        for suffix in ('mean', 'std'):
            col = f'{metric}_{suffix}'
            df[col] = pd.to_numeric(df[col], errors='raise')
            if df[col].isna().any():
                raise ValueError(f'Column {col} contains missing values.')
        if (df[f'{metric}_std'] < 0).any():
            raise ValueError(f'{metric}_std must not be negative.')
    if df.duplicated(['species', 'embedding']).any():
        raise ValueError('Expected one row per species and embedding (duplicates found).')

    species = species_order or list(dict.fromkeys(df['species']))
    unknown = set(species) - set(df['species'])
    if unknown:
        raise ValueError(f'Unknown species in --species-order: {sorted(unknown)}')
    # Both models are required to make a fair paired comparison.
    for sp in species:
        present = set(df.loc[df.species == sp, 'embedding'])
        if present != {'ProtT5', 'ESM2'}:
            raise ValueError(f'{sp}: expected both ProtT5 and ESM2, found {sorted(present)}')

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': .8,
        'savefig.facecolor': 'white',
    })
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.9), sharex=True)
    axes = axes.ravel()
    x = np.arange(len(species))
    offsets = {'ProtT5': -0.14, 'ESM2': +0.14}
    for index, ((metric, label, higher_better), ax) in enumerate(zip(METRICS, axes)):
        for emb in ('ProtT5', 'ESM2'):
            sub = df[df.embedding == emb].set_index('species').reindex(species)
            mu = sub[f'{metric}_mean'].to_numpy(float)
            sd = sub[f'{metric}_std'].to_numpy(float)
            ax.errorbar(x + offsets[emb], mu, yerr=sd, fmt='o', ms=6.3,
                        capsize=3.1, capthick=1.25, elinewidth=1.35,
                        color=PALETTE[emb], markeredgecolor='white',
                        markeredgewidth=.65, label=emb, zorder=3)
        # ax.set_title(f'{chr(65 + index)}  {label}', loc='left', weight='bold', pad=10)
        ax.set_title(f'{label}', loc='left', weight='bold', pad=10)
        ax.set_ylabel(f'{label}  ({"higher" if higher_better else "lower"} is better)')
        ax.set_xticks(x)
        ax.set_xticklabels([SPECIES.get(s, s) for s in species],
                           rotation=33, ha='right', rotation_mode='anchor')
        ax.grid(axis='y', alpha=.16, linewidth=.8)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlim(-.52, len(species)-.48)
        # Avoid a truncated zero baseline being mistaken for effect size;
        # axes are deliberately autoscaled to show subtle pairwise changes.
        vals = np.r_[df.loc[df.species.isin(species), f'{metric}_mean'].to_numpy(float)]
        errs = df.loc[df.species.isin(species), f'{metric}_std'].to_numpy(float)
        ymin = np.min(vals - errs)
        ymax = np.max(vals + errs)
        padding = max((ymax - ymin)*.17, .012)
        ax.set_ylim(ymin-padding, ymax+padding)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .995),
               frameon=False, ncol=2, columnspacing=2.5, handletextpad=.5,
               fontsize=11)
    fig.suptitle('Species-specific MIC prediction: ProtT5 vs ESM2',
                 fontsize=15, weight='bold', y=1.045)
    fig.text(.5, -.02, 'Points = mean; error bars = ±1 SD across five train–validation splits',
             ha='center', va='top', color='#555555', fontsize=10)
    fig.tight_layout(rect=[0, .005, 1, .94], h_pad=2.5, w_pad=2.0)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        destination = output / f'species_performance_comparison.{ext}'
        fig.savefig(destination, dpi=dpi, bbox_inches='tight', pad_inches=.2)
        print(f'Saved {destination}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, default=Path('validation_mean_sd.csv'), help='Input CSV with per-species means and SDs')
    p.add_argument('--output', type=Path, default=Path('species_performance_figures'), help='Output directory')
    p.add_argument('--dpi', type=int, default=600, help='PNG resolution (default: 600)')
    p.add_argument('--species-order', nargs='+', metavar='SPECIES', help='Optional species ordering (quote multi-word names)')
    args = p.parse_args()
    make_figure(pd.read_csv(args.input), args.output, args.dpi, args.species_order)


if __name__ == '__main__':
    '''
    python plot_species_performance.py \
    --input ./results/bacterial_generalization_ablation/results/validation_mean_sd.csv \
    --output ./results/bacterial_generalization_ablation/results/species_performance
    '''
    
    main()
