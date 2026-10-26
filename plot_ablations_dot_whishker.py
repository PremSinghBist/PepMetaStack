#!/usr/bin/env python3
"""Compact 2x3 PepMetaStack multi-metric ablation plots (no heatmaps).

Reads six existing CSV files from ./results/ablation_results_data by default.
Outputs PDF/SVG/PNG under ./results/plots/ablation_results.

Panels a,b,c,e,f: aligned, metric-specific horizontal dot plots. Cross-validation
standard deviations, when available, are drawn as mean +/- SD whiskers.
Panel d: changes in performance relative to the full ensemble; a negative
oriented percentage denotes worse performance, even for error metrics.

The per-metric horizontal axes are separate and are labeled with their numerical
limits; horizontal positions cannot be compared between different metrics.

Run from project root:
    python plot_ablations_dot_whishker.py
    python plot_ablations_dot_whishker.py --no-individual --dpi 600
    python plot_ablations_dot_whishker.py --data-dir path/to/csvs
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

DATA_DIR = Path('results/ablation_results_data')
OUTPUT_DIR = Path('results/plots/ablation_results')
FOUR = ('rmse', 'kendall_tau', 'r2', 'pearson_r')
SIX = ('pearson_r', 'spearman_rho', 'kendall_tau', 'rmse', 'mae', 'r2')
METRICS = {
    'rmse': (r'RMSE $\downarrow$', -1),
    'kendall_tau': (r'Kendall $\tau\uparrow$', 1),
    'r2': (r'$R^2\uparrow$', 1),
    'pearson_r': (r'Pearson $r\uparrow$', 1),
    'spearman_rho': (r'Spearman $\rho\uparrow$', 1),
    'mae': (r'MAE $\downarrow$', -1),
}
TEAL = '#087e83'
GRAY = '#758494'
RED = '#c36a55'
GRID = '#e6eaee'
DARK = '#253b49'

@dataclass(frozen=True)
class Panel:
    tag: str
    title: str
    file: str
    metrics: tuple[str, ...]
    reference: str
    aliases: dict[str, str]

PANELS = (
    Panel('a', 'PLM representations and fusion', '01_plm_representations.csv', FOUR, 'ProtT5', {
        'ESM-2 320-d': 'ESM-2 320-d', 'ESM-2 640-d': 'ESM-2 640-d',
        'ESM-2 1280-d': 'ESM-2 1280-d', 'ProtT5+ESM, Concatenation': 'ProtT5+ESM concat.',
        'ProtT5+ESM, Cross Attention': 'ProtT5+ESM cross-attn.',
        'AMP-fine-tuned ProtT5': 'AMP-tuned ProtT5'}),
    Panel('b', 'Biological feature ablation', '02_biological_features.csv', FOUR, 'PLM', {}),
    Panel('c', 'Individual base predictors', '03_base_predictors.csv', FOUR, 'PepMetaStack (Full)', {
        'PepMetaStack (Full)': 'PepMetaStack', 'HistGradientBoosting': 'HistGB'}),
    Panel('d', 'Leave-one-predictor-out', '04_leave_one_out.csv', FOUR, 'PepMetaStack (Full)', {
        'PepMetaStack (Full)': 'Full ensemble', 'w/o HistGradientBoosting': 'Remove HistGB',
        'w/o Extra Trees': 'Remove Extra Trees', 'w/o XGBoost': 'Remove XGBoost',
        'w/o Ridge': 'Remove Ridge', 'w/o MLP': 'Remove MLP',
        'w/o KNN': 'Remove KNN', 'w/o SVR': 'Remove SVR'}),
    Panel('e', 'Predictor families and ensembles', '05_predictor_families.csv', FOUR, 'PepMetaStack (Full)', {
        'PepMetaStack (Full)': 'PepMetaStack',
        'Heterogeneous-3 (SVR+ET+MLP)': 'SVR+ET+MLP',
        'Heterogeneous-2 (SVR+ET)': 'SVR+ET',
        'Distance-based (SVR+KNN)': 'SVR+KNN',
        'Tree-based (ET+HGB+XGB)': 'ET+HGB+XGB',
        'Heterogeneous-3 (Ridge+XGB+MLP)': 'Ridge+XGB+MLP'}),
    Panel('f', 'Meta-regressor comparison', '06_meta_regressors.csv', SIX, 'RidgeCV (Ours)', {
        'Linear Regression': 'Linear Reg.', 'RidgeCV (Ours)': 'RidgeCV'}),
)


def read_panel(panel: Panel, directory: Path):
    path = directory / panel.file
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; specify --data-dir if needed.')
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or 'model' not in rows[0]:
        raise ValueError(f'{path}: expected nonempty CSV with model column')
    if sum(r['model'] == panel.reference for r in rows) != 1:
        raise ValueError(f'{path}: expected exactly one reference: {panel.reference}')
    for r in rows:
        for metric in panel.metrics:
            key = metric + '_mean'
            if not r.get(key):
                raise ValueError(f'{path}: missing {key} for {r["model"]}')
            r[key] = float(r[key])
            sdkey = metric + '_sd'
            r[sdkey] = float(r[sdkey]) if r.get(sdkey) else None
    return rows


def pretty(x):
    return f'{x:.3f}' if abs(x)<1 else f'{x:.2f}'


def padded_limits(values, sds, direction, *, delta=False):
    if delta:
        size = max(0.7, np.max(np.abs(values)) * 1.2)
        return -size, size
    low = min(v - (s or 0) for v,s in zip(values,sds))
    high = max(v + (s or 0) for v,s in zip(values,sds))
    span = high-low
    if span < 0.004:
        center = (high+low)/2
        low,high = center-0.003,center+0.003
        span=high-low
    return low-0.09*span, high+0.09*span


def plot_panel(fig, spec, panel: Panel, rows, standalone=False):
    # Each panel has a dedicated model-label axis and four/six narrow metric tracks.
    n = len(rows)
    metrics=panel.metrics
    ratio = 2.60 if not standalone else 2.45
    sub = GridSpecFromSubplotSpec(1, len(metrics)+1, subplot_spec=spec,
                                 width_ratios=[ratio]+[1]*len(metrics), wspace=.16)
    lab = fig.add_subplot(sub[0,0])
    lab.set_xlim(0,1)
    lab.set_ylim(n-.45,-.55)
    lab.set_axis_off()
    # Use axes title on label axis: left aligned and without an extra title row.
    lab.set_title(f'({panel.tag}) {panel.title}', fontsize=8.7 if standalone else 8.0,
                  loc='left', fontweight='bold', color=DARK, pad=15)
    for i,row in enumerate(rows):
        isref = row['model']==panel.reference
        lab.text(.99,i,panel.aliases.get(row['model'],row['model']),
                 transform=lab.get_yaxis_transform(),ha='right',va='center',
                 fontsize=7.3 if standalone else 6.5,
                 color=TEAL if isref else DARK,
                 fontweight='bold' if isref else 'normal')
    ref = next(r for r in rows if r['model']==panel.reference)
    for k,metric in enumerate(metrics):
        ax=fig.add_subplot(sub[0,k+1],sharey=lab)
        direction=METRICS[metric][1]
        means=np.array([r[metric+'_mean'] for r in rows])
        sds=[r[metric+'_sd'] for r in rows]
        is_delta = panel.tag == 'd'
        if is_delta:
            # Positive is better; the same percent delta scale for all tracks.
            values=direction*(means-ref[metric+'_mean'])/abs(ref[metric+'_mean'])*100
            # Reference is exactly zero. SD is deliberately not propagated here:
            # paired fold-level covariances are unavailable in the CSV.
            lo,hi=padded_limits(values,[None]*n,direction,delta=True)
            ax.axvline(0,lw=.85,c='#536e7a',zorder=1)
            ax.hlines(np.arange(n), 0, values, color=[TEAL if v>0 else RED if v<0 else GRID for v in values],lw=1.25,zorder=2)
            ax.scatter(values,np.arange(n),s=16,c=[TEAL if v>0 else RED if v<0 else GRAY for v in values],zorder=3,
                       edgecolors='white',linewidths=.35)
        else:
            values=means
            lo,hi=padded_limits(means,sds,direction)
            ax.axvline(ref[metric+'_mean'],c=TEAL,ls=':',lw=.85,alpha=.65,zorder=1)
            for i,(v,sd,row) in enumerate(zip(means,sds,rows)):
                color=TEAL if row['model']==panel.reference else GRAY
                if sd is not None and sd>0:
                    ax.errorbar(v,i,xerr=sd,fmt='none',elinewidth=.95,ecolor=color,
                                capsize=1.6,capthick=.8,zorder=2,alpha=.85)
                ax.scatter(v,i,s=24 if color==TEAL else 15,color=color,edgecolors='white',
                           linewidths=.38,zorder=3)
        ax.set_xlim(lo,hi)
        ax.set_ylim(n-.45,-.55)
        ax.set_yticks([])
        ax.set_xticks([])
        if is_delta:
            range_label=f'{lo:.1f} to +{hi:.1f}%'
        else:
            range_label=f'{pretty(lo)}–{pretty(hi)}'
        ax.set_xlabel(range_label,fontsize=5.2 if standalone else 4.8,
                      color='#536776',labelpad=2)
        short={'pearson_r':r'$r\uparrow$', 'spearman_rho':r'$\rho\uparrow$',
               'kendall_tau':r'$\tau\uparrow$', 'r2':r'$R^2\uparrow$',
               'rmse':r'RMSE$\downarrow$', 'mae':r'MAE$\downarrow$'}
        head=short[metric] if len(metrics)==6 else METRICS[metric][0]
        ax.set_title(head,fontsize=6.8 if standalone else 6.0,
                     color=DARK,pad=5)
        for i in range(n):
            ax.axhline(i,color=GRID,lw=.42,zorder=0)
        for sp in ('top','left','right'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_linewidth(.55)
        ax.spines['bottom'].set_color('#9aaab2')


def save(fig, outdir, stem, dpi):
    for ext in ('pdf','svg','png'):
        path=outdir/f'{stem}.{ext}'
        fig.savefig(path,dpi=dpi if ext=='png' else None,facecolor='white')
        print('Saved',path)


def combined(data,outdir,dpi,width,height):
    fig=plt.figure(figsize=(width,height),facecolor='white')
    grid=fig.add_gridspec(3,2,left=.018,right=.988,top=.956,bottom=.087,
                         hspace=.32,wspace=.065,height_ratios=[9,8,7])
    for i,(p,rows) in enumerate(data):
        plot_panel(fig,grid[i//2,i%2],p,rows)
    handles=[Line2D([0],[0],marker='o',lw=0,markerfacecolor=TEAL,markeredgecolor='white',markersize=6,label='Reference'),
             Line2D([0],[0],marker='o',lw=0,markerfacecolor=GRAY,markeredgecolor='white',markersize=6,label='Alternative'),
             Line2D([0],[0],color=GRAY,marker='|',lw=1,markersize=8,label='Mean ± SD')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=3,
               fontsize=7,frameon=False,columnspacing=2)
    fig.text(.5,.015,'Each metric uses its own labeled scale. Panel (d): directional % change relative to full ensemble (negative = worse).',
             ha='center',va='center',fontsize=6.7,color='#50606a')
    save(fig,outdir,'pepmetastack_ablations_compact_2x3',dpi)
    plt.close(fig)


def standalone(data,outdir,dpi):
    for panel,rows in data:
        fig=plt.figure(figsize=(8.1, 2.65 if len(rows)>=8 else 2.40),facecolor='white')
        spec=fig.add_gridspec(1,1,left=.015,right=.985,top=.84,bottom=.18)
        plot_panel(fig,spec[0],panel,rows,standalone=True)
        save(fig,outdir,panel.file.replace('.csv','_compact'),dpi)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT_DIR)
    parser.add_argument('--dpi',type=int,default=600)
    parser.add_argument('--width',type=float,default=9.2,help='Combined figure width (inches)')
    parser.add_argument('--height',type=float,default=9.0,help='Combined figure height (inches)')
    parser.add_argument('--no-individual',action='store_true')
    args=parser.parse_args()
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.unicode_minus':False})
    data=[(p,read_panel(p,args.data_dir)) for p in PANELS]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    combined(data,args.output_dir,args.dpi,args.width,args.height)
    if not args.no_individual:
        standalone(data,args.output_dir,args.dpi)
    print('Done. 6 CSVs read; no heatmaps or CSV generation required.')

if __name__ == '__main__':
    main()
