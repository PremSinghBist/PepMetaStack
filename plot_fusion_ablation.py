#!/usr/bin/env python3
"""Publication-style Matplotlib figures from fusion ablation CSV exports.

Usage:
  python plot_fusion_ablation.py --input-dir ./results/ecoli_fusion_ablation \
      --output-dir ./results/ecoli_fusion_ablation/figures

Required: numpy, pandas, matplotlib. Optional: scipy (not needed).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # works on servers without displays
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METRICS = ['Pearson_R', 'Spearman_R', 'Kendall_Tau', 'RMSE', 'MAE', 'R2']
NAMES = {'Pearson_R':'Pearson R', 'Spearman_R':'Spearman R',
         'Kendall_Tau':'Kendall τ', 'RMSE':'RMSE', 'MAE':'MAE', 'R2':'R²'}
PREFERRED = ['concat','attention','cross_attention','gated']
LABELS = {'concat':'Concatenation','attention':'Attention',
          'cross_attention':'Cross-attention','gated':'Gated fusion'}
PALETTE = {'concat':'#277da1','attention':'#f4a261',
           'cross_attention':'#bc6c8a','gated':'#40916c'}


def color(method):
    return PALETTE.get(method, '#777777')


def ordered_methods(methods):
    methods = set(methods)
    return [m for m in PREFERRED if m in methods] + sorted(methods-set(PREFERRED))


def save(fig, out, stem, formats, dpi):
    fig.tight_layout(pad=2)
    for ext in formats:
        path = out / f'{stem}.{ext}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print('Saved:', path)
    plt.close(fig)


def require_columns(df, expected, filename):
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f'{filename}: missing required columns: {sorted(missing)}')


def comparison_table(report, seed_summary, folds):
    """Return method x metric mean and error matrices, with correct SD scope.

    One seed => across-five-validation-fold SD, from seed_summary.
    Multiple seeds => across-seed SD of five-fold means, from report.
    """
    n_seeds = folds['Seed'].nunique()
    if n_seeds == 1:
        if len(seed_summary) == 0:
            raise ValueError('Seed summary required for one-seed fold SD bars.')
        grouped = seed_summary.drop_duplicates('Fusion').set_index('Fusion')
        means = pd.DataFrame({m: grouped[f'{m}_Mean'] for m in METRICS})
        errors = pd.DataFrame({m: grouped[f'{m}_Std'] for m in METRICS})
        legend = 'Error bars: ± 1 SD across 5 validation folds (one seed)'
    else:
        means = report.pivot(index='Fusion',columns='Metric',values='Mean')
        errors = report.pivot(index='Fusion',columns='Metric',values='Std')
        legend = f'Error bars: ± 1 SD across {n_seeds} seed-level CV means'
    methods = ordered_methods(folds['Fusion'].unique())
    return means.loc[methods, METRICS], errors.loc[methods, METRICS], legend


def plot_comparison(means, errors, err_label, out, formats, dpi):
    methods = list(means.index)
    fig, axes = plt.subplots(2,3, figsize=(16,9))
    for ax, m in zip(axes.flat, METRICS):
        x = np.arange(len(methods))
        vals = means[m].to_numpy(float)
        err = np.nan_to_num(errors[m].to_numpy(float), nan=0.)
        ax.bar(x, vals, yerr=err, capsize=4, color=[color(k) for k in methods],
               edgecolor='white', linewidth=.7, error_kw={'elinewidth':1.1})
        ax.set_xticks(x, [LABELS.get(k,k) for k in methods], rotation=20, ha='right')
        ax.set_title(NAMES[m] + (' ↑' if m not in ('RMSE','MAE') else ' ↓'),fontweight='bold')
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
        span=max(np.max(vals+err)-np.min(vals-err), .015)
        ax.set_ylim(max(0,np.min(vals-err)-span*.45),np.max(vals+err)+span*.45)
        for j,v in enumerate(vals):
            ax.text(j,v+err[j]+span*.04,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
    fig.suptitle('Feature-fusion ablation — validation performance',fontsize=17,fontweight='bold')
    fig.text(.5,.015,err_label,ha='center',fontsize=10)
    fig.subplots_adjust(bottom=.09)
    save(fig,out,'01_all_metrics_mean_sd',formats,dpi)


def plot_fold_trends(folds, out, formats, dpi):
    methods=ordered_methods(folds['Fusion'].unique())
    seeds=sorted(folds['Seed'].unique())
    fig,axs=plt.subplots(2,3,figsize=(16,9),sharex=True)
    for ax,m in zip(axs.flat,METRICS):
        for method in methods:
            d=folds[folds['Fusion']==method]
            agg=d.groupby('Fold')[m].agg(['mean','std']).sort_index()
            xx=agg.index.to_numpy()
            y=agg['mean'].to_numpy()
            ax.plot(xx,y,marker='o',lw=2,color=color(method),label=LABELS.get(method,method))
            if len(seeds)>1:
                e=agg['std'].fillna(0).to_numpy()
                ax.fill_between(xx,y-e,y+e,color=color(method),alpha=.1)
        ax.set_title(NAMES[m],fontweight='bold')
        ax.set_xticks(sorted(folds['Fold'].unique()))
        ax.set_xlabel('Outer validation fold')
        ax.grid(alpha=.2)
    handles=[Line2D([0],[0],color=color(k),marker='o',lw=2,label=LABELS.get(k,k)) for k in methods]
    fig.legend(handles=handles,loc='lower center',ncol=min(4,len(handles)),bbox_to_anchor=(.5,0.005),frameon=False)
    fig.suptitle('Performance across validation folds' + (' (averaged over seeds)' if len(seeds)>1 else ''),fontsize=17,fontweight='bold')
    fig.subplots_adjust(bottom=.12)
    save(fig,out,'02_fold_trends',formats,dpi)


def plot_fold_boxplots(folds,out,formats,dpi):
    methods=ordered_methods(folds['Fusion'].unique())
    fig,axs=plt.subplots(2,3,figsize=(16,9))
    for ax,m in zip(axs.flat,METRICS):
        datasets=[folds.loc[folds['Fusion']==k,m].dropna().to_numpy() for k in methods]
        bp=ax.boxplot(datasets,patch_artist=True,showmeans=True,
                      meanprops=dict(marker='D',markerfacecolor='black',markeredgecolor='black',markersize=4))
        for box,method in zip(bp['boxes'],methods):
            box.set_facecolor(color(method)); box.set_alpha(.67)
        ax.set_xticks(range(1,len(methods)+1),[LABELS.get(k,k) for k in methods],rotation=20,ha='right')
        ax.set_title(NAMES[m],fontweight='bold')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Distribution of validation-fold scores',fontsize=17,fontweight='bold')
    save(fig,out,'03_fold_distributions',formats,dpi)


def plot_rank_heatmap(means,out,formats,dpi):
    methods=list(means.index)
    ranks=means.rank(axis=0,ascending=True,method='min')
    for m in ['Pearson_R','Spearman_R','Kendall_Tau','R2']:
        ranks[m]=means[m].rank(ascending=False,method='min')
    # RMSE/MAE remain ascending=True. Lower rank is better.
    fig,ax=plt.subplots(figsize=(11,4.5))
    im = ax.imshow(ranks.to_numpy(), cmap='YlGnBu_r', vmin=1, vmax=max(len(methods), 2), aspect='auto')
    ax.set_xticks(range(len(METRICS)),[NAMES[m] for m in METRICS])
    ax.set_yticks(range(len(methods)),[LABELS.get(k,k) for k in methods])
    for i in range(len(methods)):
        for j,m in enumerate(METRICS):
            ax.text(j,i,f'#{int(ranks.iloc[i,j])}\n{means.iloc[i,j]:.4f}',ha='center',va='center',fontsize=9)
    ax.set_title('Rank by metric (1 = best) and corresponding mean',fontweight='bold')
    fig.colorbar(im,ax=ax,label='Rank (lower is better)',ticks=range(1,len(methods)+1))
    save(fig,out,'04_metric_rank_heatmap',formats,dpi)


def plot_seed_trends(seed_summary,out,formats,dpi):
    if seed_summary['Seed'].nunique()<2:
        print('Skipped seed trends: only one seed in data.')
        return
    methods=ordered_methods(seed_summary['Fusion'].unique())
    fig,axs=plt.subplots(2,3,figsize=(16,9))
    for ax,m in zip(axs.flat,METRICS):
        for method in methods:
            d=seed_summary.loc[seed_summary['Fusion']==method].sort_values('Seed')
            ax.plot(d['Seed'].astype(str),d[f'{m}_Mean'],marker='o',label=LABELS.get(method,method),color=color(method))
        ax.set_title(NAMES[m],fontweight='bold');ax.grid(alpha=.2)
        ax.tick_params(axis='x',labelrotation=45)
    handles=[Line2D([0],[0],color=color(k),marker='o',lw=2,label=LABELS.get(k,k)) for k in methods]
    fig.legend(handles=handles,loc='lower center',ncol=min(4,len(handles)),frameon=False)
    fig.suptitle('Five-fold CV mean performance across random seeds',fontsize=17,fontweight='bold')
    save(fig,out,'05_seed_sensitivity',formats,dpi)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir',type=Path,default=Path('.'))
    p.add_argument('--output-dir',type=Path,default=Path('fusion_figures'))
    p.add_argument('--formats',nargs='+',choices=['png','pdf','svg'],default=['png','pdf'])
    p.add_argument('--dpi',type=int,default=300)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    prefix='fusion_ablation_validation_'
    folds=pd.read_csv(a.input_dir/(prefix+'fold_metrics.csv'))
    seeds=pd.read_csv(a.input_dir/(prefix+'seed_summary.csv'))
    report=pd.read_csv(a.input_dir/(prefix+'report.csv'))
    require_columns(folds,['Seed','Fusion','Fold']+METRICS,'fold_metrics.csv')
    require_columns(seeds,['Seed','Fusion']+[f'{m}_{s}' for m in METRICS for s in ('Mean','Std')],'seed_summary.csv')
    require_columns(report,['Fusion','Metric','Mean','Std'],'report.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':a.dpi,'axes.titlepad':10,'figure.facecolor':'white'})
    means,errors,err_label=comparison_table(report,seeds,folds)
    plot_comparison(means,errors,err_label,a.output_dir,a.formats,a.dpi)
    plot_fold_trends(folds,a.output_dir,a.formats,a.dpi)
    plot_fold_boxplots(folds,a.output_dir,a.formats,a.dpi)
    plot_rank_heatmap(means,a.output_dir,a.formats,a.dpi)
    plot_seed_trends(seeds,a.output_dir,a.formats,a.dpi)
    print('\nPlotting complete. Error convention:',err_label)

if __name__=='__main__':
    main()
