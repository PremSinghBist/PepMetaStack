#!/usr/bin/env python3
""" 2x2 E. coli AMP sequence / MIC analysis.

Panels
------
(a) Peptide length distribution and median / interquartile range.
(b) Amino-acid abundance across all observed peptide residues.
(c) Peptide length versus activity endpoint, with bootstrap CIs of binned medians.
(d) Per-residue length-adjusted partial Spearman associations with the activity
    endpoint, with bootstrap CIs and Benjamini-Hochberg FDR adjustment.

CSV outputs include cleaned peptide-level data, sequence-level QC (Quality control), residue
abundance/prevalence, length-binned activity, unadjusted and length-adjusted
residue associations, and activity-quartile composition.

Important interpretation
------------------------
--activity-transform none treats 'value' EXACTLY as provided. Use 'log10' only
if the supplied values are raw positive MIC measurements; DO NOT log-transform
an already log10-transformed endpoint. Positive correlations with raw MIC or
log10 MIC indicate larger MIC (weaker antimicrobial potency), not stronger.
These are observational associations, not causal residue contributions.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

AA = tuple('ACDEFGHIKLMNPQRSTVWY')
AA_SET = set(AA)
COLOR = '#20668D'
OTHER = '#7994A5'
ACCENT = '#B16C54'
PALE = '#D8E6EE'
DARK = '#243746'


def read_and_clean(path: Path, duplicate_policy: str, activity_transform: str):
    df = pd.read_csv(path)
    required = {'sequence', 'value'}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {sorted(required)}, got {list(df.columns)}")
    raw_count = len(df)
    sequences = df['sequence'].astype('string').str.upper().str.replace(r'\s+', '', regex=True)
    valid = sequences.notna() & sequences.str.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]+').fillna(False)
    invalid_count = int((~valid).sum())
    if invalid_count:
        warnings.warn(f'Dropping {invalid_count} invalid/ambiguous/missing sequences (only 20 standard residues allowed).')
    df = df.loc[valid, ['value']].copy()
    df.insert(0, 'sequence', sequences.loc[valid].astype(str).values)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.loc[~np.isfinite(df['value'].to_numpy(float)), 'value'] = np.nan
    if df.empty:
        raise ValueError('No valid peptide sequences remain after validation.')
    unique_before = df.sequence.nunique()
    duplicates_count = len(df) - unique_before
    if duplicate_policy == 'error' and duplicates_count:
        raise ValueError(f'{duplicates_count} repeated sequence rows found; choose --duplicate-policy median or keep')
    if duplicate_policy == 'median':
        # Work at the peptide-sequence level. Identical sequences in multiple
        # CV-fold exports should not inflate frequencies / correlations.
        grouped = df.groupby('sequence', sort=False, observed=True)['value']
        seqdf = grouped.agg(activity='median', n_records='size', n_activity='count',
                            activity_min='min', activity_max='max').reset_index()
    elif duplicate_policy == 'keep':
        warnings.warn('Keeping repeated sequences treats repeated rows as independent; consider median for CV exports.')
        seqdf = df.rename(columns={'value': 'activity'}).copy()
        seqdf['n_records'] = 1
        seqdf['n_activity'] = seqdf.activity.notna().astype(int)
        seqdf['activity_min'] = seqdf.activity
        seqdf['activity_max'] = seqdf.activity
    else:
        raise ValueError(f'Unsupported duplicate policy: {duplicate_policy}')
    before_transform = seqdf['activity'].to_numpy(dtype=float)
    if activity_transform in ('log10', 'neglog10'):
        bad = np.isfinite(before_transform) & (before_transform <= 0)
        if bad.any():
            raise ValueError(f'Cannot log10-transform {int(bad.sum())} nonpositive MIC values. Is value already log-transformed?')
        transformed = np.where(np.isfinite(before_transform), np.log10(before_transform), np.nan)
        if activity_transform == 'neglog10':
            transformed = -transformed
        seqdf['activity'] = transformed
    elif activity_transform != 'none':
        raise ValueError(activity_transform)
    seqdf['length'] = seqdf['sequence'].str.len().astype(int)
    seqdf['activity_is_finite'] = np.isfinite(seqdf['activity'].to_numpy(float))
    for residue in AA:
        counts = seqdf['sequence'].str.count(residue).astype(int)
        seqdf[f'count_{residue}'] = counts
        seqdf[f'frac_{residue}'] = counts / seqdf['length']
    metadata = {
        'input_rows': raw_count, 'invalid_sequences_removed': invalid_count,
        'valid_input_rows': len(df), 'unique_peptide_sequences': unique_before,
        'repeated_sequence_rows': duplicates_count, 'analyzed_rows': len(seqdf),
        'analyzed_rows_with_activity': int(seqdf.activity_is_finite.sum()),
        'analyzed_rows_missing_activity': int((~seqdf.activity_is_finite).sum()),
        'min_length': int(seqdf.length.min()), 'median_length': float(seqdf.length.median()),
        'mean_length': float(seqdf.length.mean()), 'max_length': int(seqdf.length.max()),
        'length_q1': float(seqdf.length.quantile(.25)),
        'length_q3': float(seqdf.length.quantile(.75)),
        'duplicate_policy': duplicate_policy, 'activity_transform': activity_transform,
    }
    if duplicates_count and duplicate_policy == 'median':
        different = seqdf[(seqdf.n_activity > 1) & (seqdf.activity_min != seqdf.activity_max)]
        metadata['unique_sequences_with_discordant_values'] = len(different)
        if len(different):
            warnings.warn(f'{len(different)} peptide sequences have multiple distinct endpoint values; using per-sequence median.')
    return seqdf, metadata


def frequencies(seqdf: pd.DataFrame):
    length_sum = int(seqdf.length.sum())
    n = len(seqdf)
    rows = []
    for aa in AA:
        counts = seqdf[f'count_{aa}'].to_numpy(dtype=float)
        frac = seqdf[f'frac_{aa}'].to_numpy(dtype=float)
        rows.append({
            'residue': aa,
            'total_occurrences': int(counts.sum()),
            'total_observed_residues': length_sum,
            'pooled_frequency_percent': 100.0 * counts.sum() / length_sum,
            'mean_per_peptide_fraction_percent': 100.0 * frac.mean(),
            'peptide_prevalence_percent': 100.0 * np.count_nonzero(counts) / n,
            'peptides_containing_residue': int(np.count_nonzero(counts)),
            'n_peptides': n,
        })
    return pd.DataFrame(rows).sort_values('pooled_frequency_percent', ascending=False).reset_index(drop=True)


def spearman_safe(x, y):
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    stat = stats.spearmanr(x, y)
    return float(stat.statistic), float(stat.pvalue)


def partial_ranks(X: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Partial Spearman: correlate ranked-X and ranked-y after removing rank(z)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    RX = stats.rankdata(X, axis=0, method='average')
    RY = stats.rankdata(y, method='average')
    RZ = stats.rankdata(z, method='average')
    covariate = np.column_stack([np.ones(len(RZ)), RZ])
    # Least squares handles repeated length ranks; if constant length, use
    # intercept-only adjustment for more stable results.
    if np.ptp(RZ) <= 1e-12:
        covariate = covariate[:, :1]
    RX_res = RX - covariate @ np.linalg.lstsq(covariate, RX, rcond=None)[0]
    RY_res = RY - covariate @ np.linalg.lstsq(covariate, RY, rcond=None)[0]
    RX_res = RX_res - RX_res.mean(axis=0)
    RY_res = RY_res - RY_res.mean()
    numer = RX_res.T @ RY_res
    denom = np.sqrt(np.sum(RX_res ** 2, axis=0) * np.sum(RY_res ** 2))
    return np.divide(numer, denom, out=np.full(X.shape[1], np.nan), where=denom > 1e-10)


def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    p0 = p[ok]
    if not len(p0):
        return q
    order = np.argsort(p0)
    ranks = np.arange(1, len(p0) + 1)
    ordered = np.minimum.accumulate((p0[order] * len(p0) / ranks)[::-1])[::-1]
    result = np.empty(len(p0))
    result[order] = np.minimum(1.0, ordered)
    q[ok] = result
    return q


def associations(seqdf, bootstraps: int, seed: int):
    active = seqdf.loc[seqdf.activity_is_finite].copy()
    n = len(active)
    if n < 8:
        raise ValueError('Need at least 8 peptides with finite activity values for association analysis.')
    X = active[[f'frac_{a}' for a in AA]].to_numpy(dtype=float)
    y = active.activity.to_numpy(dtype=float)
    L = active.length.to_numpy(dtype=float)
    raw = np.array([spearman_safe(X[:, j], y) for j in range(20)])
    partial = partial_ranks(X, y, L)
    # Approximate analytic p-values (t-distribution, df=n-3). Not a causal test;
    # p-values assume independent peptides. Fold membership is not a sample unit.
    dfree = n - 3
    partial_p = np.where(np.isfinite(partial) & (np.abs(partial) < 1),
                         2 * stats.t.sf(np.abs(partial) * np.sqrt(dfree / np.maximum(1e-15, 1 - partial**2)), dfree),
                         np.nan)
    CI = np.full((20, 2), np.nan)
    if bootstraps > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty((bootstraps, 20), dtype=float)
        for b in range(bootstraps):
            sel = rng.integers(0, n, size=n)
            sims[b] = partial_ranks(X[sel], y[sel], L[sel])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            CI[:, 0] = np.nanpercentile(sims, 2.5, axis=0)
            CI[:, 1] = np.nanpercentile(sims, 97.5, axis=0)
    q = benjamini_hochberg(partial_p)
    out = pd.DataFrame({
        'residue': AA,
        'n_peptides': n,
        'spearman_rho_raw': raw[:, 0],
        'spearman_p_raw': raw[:, 1],
        'partial_spearman_rho_adjusted_length': partial,
        'partial_spearman_p_approx': partial_p,
        'partial_spearman_fdr_bh_q': q,
        'partial_spearman_bootstrap_ci_lower_95': CI[:, 0],
        'partial_spearman_bootstrap_ci_upper_95': CI[:, 1],
        'fdr_significant_0_05': q < 0.05,
        'bootstrap_replicates': bootstraps,
    })
    length_rho, length_p = spearman_safe(L, y)
    length_relation = pd.DataFrame([{
        'n_peptides': n, 'spearman_rho_length_vs_endpoint': length_rho,
        'spearman_p_length_vs_endpoint': length_p,
        'endpoint_median': float(np.median(y)),
        'endpoint_mean': float(np.mean(y)),
        'length_median': float(np.median(L)),
    }])
    return out.sort_values('partial_spearman_rho_adjusted_length', ascending=True).reset_index(drop=True), length_relation


def median_bootstrap_ci(x, rng, B):
    if not len(x):
        return np.nan, np.nan
    if len(x) < 4 or B == 0:
        return np.nan, np.nan
    # Vectorized bootstrap for a small bin: avoid a huge intermediate matrix
    values = np.empty(B, dtype=float)
    for i in range(B):
        values[i] = np.median(x[rng.integers(0, len(x), len(x))])
    return tuple(np.percentile(values, [2.5, 97.5]))


def length_bins(seqdf, n_bins, bootstraps, seed):
    active = seqdf.loc[seqdf.activity_is_finite, ['length', 'activity']].copy()
    if len(active) < 8:
        return pd.DataFrame()
    max_quantiles = min(n_bins, active.length.nunique(), len(active) // 4)
    if max_quantiles < 2:
        return pd.DataFrame()
    active['bin'] = pd.qcut(active.length, q=max_quantiles, duplicates='drop')
    rng = np.random.default_rng(seed + 31)
    rows = []
    for interval, group in active.groupby('bin', observed=True, sort=True):
        yy = group.activity.to_numpy(dtype=float)
        lo, hi = median_bootstrap_ci(yy, rng, bootstraps)
        rows.append({
            'length_bin': str(interval), 'length_min': int(group.length.min()),
            'length_max': int(group.length.max()), 'length_median': float(group.length.median()),
            'n_peptides': len(group), 'activity_mean': float(np.mean(yy)),
            'activity_median': float(np.median(yy)),
            'activity_median_ci_lower_95': lo,
            'activity_median_ci_upper_95': hi,
        })
    return pd.DataFrame(rows)


def length_distribution(seqdf):
    return (seqdf.groupby('length', observed=True).size().rename('n_peptides').reset_index()
            .sort_values('length').reset_index(drop=True))


def activity_groups(seqdf):
    active = seqdf.loc[seqdf.activity_is_finite].copy()
    if len(active) < 8 or active.activity.nunique() < 4:
        return pd.DataFrame()
    # Equal-frequency activity quartiles, where possible. These are endpoint
    # quartiles, NOT potency categories unless the endpoint direction is known.
    try:
        active['activity_quartile'] = pd.qcut(active.activity, q=4, labels=False, duplicates='drop')
    except ValueError:
        return pd.DataFrame()
    rows=[]
    for k, sub in active.groupby('activity_quartile', observed=True, sort=True):
        for aa in AA:
            rows.append({'activity_quartile_from_low_to_high': int(k) + 1,
                         'residue': aa, 'n_peptides': len(sub),
                         'mean_residue_fraction_percent': float(100*sub[f'frac_{aa}'].mean()),
                         'pooled_fraction_percent': float(100*sub[f'count_{aa}'].sum()/sub.length.sum()),
                         'median_activity': float(sub.activity.median())})
    return pd.DataFrame(rows)


def axis_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=7, direction='out', length=2.6, width=0.6)
    ax.grid(axis='y', color='#DDE5EA', lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def make_plot(seqdf, freq, len_bins, associations_df, relation, args):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.dpi': args.dpi,
    })
    fig, axs = plt.subplots(2, 2, figsize=(args.width, args.height),
                            gridspec_kw={'height_ratios': [0.94, 1.06]})
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.115, top=0.92,
                        wspace=0.37, hspace=0.49)
    a, b, c, d = axs.flat
    for ax in axs.flat:
        axis_style(ax)

    # (a) Peptide length histogram, unweighted by repeated CV-fold records.
    lengths = seqdf.length.to_numpy()
    hist_bins = np.histogram_bin_edges(lengths, bins='fd') if len(np.unique(lengths))>1 else np.array([lengths[0]-.5,lengths[0]+.5])
    if len(hist_bins) < 9:
        hist_bins = np.linspace(lengths.min() - .5, lengths.max() + .5, min(15, max(3, lengths.max()-lengths.min()+2)))
    if len(hist_bins) > 37:
        hist_bins = np.linspace(lengths.min() - .5, lengths.max() + .5, 36)
    a.hist(lengths, bins=hist_bins, color=COLOR, edgecolor='white', linewidth=.45)
    med = np.median(lengths)
    a.axvline(med, ls='--', lw=1.0, color=ACCENT, label=f'Median = {med:g} aa')
    a.legend(frameon=False, fontsize=6.6, loc='upper right', handlelength=1.4)
    a.set_xlabel('Sequence length (amino acids)')
    a.set_ylabel('Number of peptides')
    a.set_title('Length distribution', loc='left', weight='bold', pad=5)

    # (b) Overall pooled amino-acid relative abundance.
    top = freq.copy()
    xx = np.arange(len(top))
    colors = [ACCENT if i < 3 else OTHER for i in range(len(top))]
    b.bar(xx, top.pooled_frequency_percent, color=colors, width=.79, zorder=3)
    b.set_xticks(xx)
    b.set_xticklabels(top.residue, fontsize=6.4)
    b.set_ylabel('Residues in dataset (%)')
    b.set_xlabel('Amino acid (descending abundance)')
    b.set_title('Residue frequency', loc='left', weight='bold', pad=5)
    b.set_xlim(-.7, len(top)-.3)
    b.tick_params(axis='x', length=0)

    # (c) Peptide-level length and value relationship with binned median + CI.
    active = seqdf.loc[seqdf.activity_is_finite]
    if len(active) >= 4:
        xall = active.length.to_numpy(dtype=float)
        yall = active.activity.to_numpy(dtype=float)
        # Use deterministic subsample only for point plotting; all data enter stats.
        if len(active) > 6500:
            pos = np.random.default_rng(args.seed + 97).choice(len(active), 6500, replace=False)
            xplot, yplot = xall[pos], yall[pos]
        else:
            xplot, yplot = xall, yall
        c.scatter(xplot, yplot, s=5.5, facecolor=OTHER, alpha=.19, edgecolor='none', zorder=2)
        if not len_bins.empty:
            xm = len_bins.length_median.to_numpy(dtype=float)
            ym = len_bins.activity_median.to_numpy(dtype=float)
            good = np.isfinite(len_bins.activity_median_ci_lower_95) & np.isfinite(len_bins.activity_median_ci_upper_95)
            c.plot(xm, ym, '-', color=ACCENT, lw=1.2, zorder=5)
            if good.any():
                low = ym[good] - len_bins.loc[good, 'activity_median_ci_lower_95'].to_numpy(float)
                high = len_bins.loc[good, 'activity_median_ci_upper_95'].to_numpy(float) - ym[good]
                c.errorbar(xm[good], ym[good], yerr=np.vstack([np.maximum(low,0),np.maximum(high,0)]),
                           fmt='o', markersize=3.3, capsize=2, color=ACCENT,
                           elinewidth=.85, zorder=6, label='Binned median ± 95% CI')
            else:
                c.plot(xm, ym, 'o', markersize=3.4, color=ACCENT, label='Binned median')
            c.legend(frameon=False, fontsize=6.1, loc='best')
        rho = relation.iloc[0].spearman_rho_length_vs_endpoint
        if np.isfinite(rho):
            c.text(.03, .97, rf'Spearman $\rho$ = {rho:+.3f}', transform=c.transAxes,
                   va='top', fontsize=7,
                   bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':0.85, 'pad':1.5})
    else:
        c.text(.5, .5, 'Insufficient activity data', ha='center', va='center', transform=c.transAxes)
    c.set_xlabel('Sequence length (amino acids)')
    ylabels = {'none':'Activity endpoint (value)', 'log10':r'$\log_{10}(\mathrm{MIC})$',
               'neglog10':r'$-\log_{10}(\mathrm{MIC})$'}
    c.set_ylabel(ylabels[args.activity_transform])
    c.set_title('Length and MIC endpoint', loc='left', weight='bold', pad=5)

    # (d) All 20 residue effect sizes: length-adjusted partial rank correlations.
    x = associations_df.partial_spearman_rho_adjusted_length.to_numpy(float)
    low = associations_df.partial_spearman_bootstrap_ci_lower_95.to_numpy(float)
    high = associations_df.partial_spearman_bootstrap_ci_upper_95.to_numpy(float)
    sign = associations_df.fdr_significant_0_05.to_numpy(bool)
    yy = np.arange(len(x))
    d.axvline(0, color='#34495E', lw=.8, zorder=2)
    for i in range(len(x)):
        if np.isfinite(low[i]) and np.isfinite(high[i]):
            d.hlines(i, low[i], high[i], color=COLOR if sign[i] else '#AABBC6', lw=1.25, zorder=3)
        if np.isfinite(x[i]):
            d.plot(x[i], i, 'o', ms=3.3, color=COLOR if sign[i] else OTHER,
                   markeredgecolor='white', markeredgewidth=.4, zorder=5)
    d.set_yticks(yy)
    d.set_yticklabels([f'{r}*' if sig else r for r,sig in zip(associations_df.residue,sign)], fontsize=6.3)
    d.set_ylim(-.9, len(yy)-.1)
    d.set_xlabel(r'Length-adjusted partial Spearman $\rho$')
    d.set_title('Residue composition vs. MIC endpoint', loc='left', weight='bold', pad=5)
    d.grid(axis='x', color='#DDE5EA', lw=.45, zorder=0)
    d.grid(axis='y', visible=False)
    fig.text(.98, .043, '* BH-FDR q < 0.05 (residue associations)',
             ha='right', va='center', fontsize=6.6, color=DARK)
    xs = np.concatenate([x[np.isfinite(x)],low[np.isfinite(low)],high[np.isfinite(high)],np.array([0.])])
    bound = max(.05,float(np.max(np.abs(xs))) * 1.17)
    d.set_xlim(-bound, bound)

    for tag, ax in zip('abcd', axs.flat):
        ax.text(-.11, 1.085, f'({tag})', transform=ax.transAxes,
                fontsize=9, fontweight='bold', ha='left', va='bottom')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ('png', 'pdf', 'svg'):
        fp = args.output_dir / f'{args.filename}.{ext}'
        fig.savefig(fp, dpi=args.dpi, facecolor='white', bbox_inches='tight', pad_inches=.075)
        paths.append(fp)
    plt.close(fig)
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input',type=Path,default=Path('data/ecoli_five_fold_dataset.csv'))
    p.add_argument('--output-dir',type=Path,default=Path('plot/pep_analysis'))
    p.add_argument('--filename',default='ecoli_peptide_sequence_activity_2x2')
    p.add_argument('--duplicate-policy',choices=['median','keep','error'],default='median',
                   help='How to handle repeated peptide sequences (median is safest for fold-repeated CSVs).')
    p.add_argument('--activity-transform',choices=['none','log10','neglog10'],default='none',
                   help='none preserves value; log10 assumes positive RAW MIC; never log an already logged endpoint.')
    p.add_argument('--bins',type=int,default=8,help='Target number of activity-vs-length quantile bins.')
    p.add_argument('--bootstraps',type=int,default=300,
                   help='Bootstrap iterations for 95%% CIs of length-binned medians and residue associations.')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--width',type=float,default=7.1,help='Figure width in inches for double-column layout.')
    p.add_argument('--height',type=float,default=5.9,help='Figure height in inches.')
    p.add_argument('--dpi',type=int,default=600)
    args=p.parse_args()
    if not args.input.is_file():
        p.error(f'Input CSV not found: {args.input}. Pass --input /path/to/ecoli_five_fold_dataset.csv')
    if args.bins < 2 or args.bootstraps < 0 or args.width <= 0 or args.height <= 0 or args.dpi < 72:
        p.error('Require --bins >= 2, --bootstraps >= 0, positive width/height, --dpi >= 72.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pep,qc=read_and_clean(args.input,args.duplicate_policy,args.activity_transform)
    if pep.activity_is_finite.sum() < 8:
        p.error('Need at least 8 unique peptides with non-missing MIC endpoint for four-panel analysis.')
    fq=frequencies(pep)
    len_df=length_distribution(pep)
    assoc,relation=associations(pep,args.bootstraps,args.seed)
    bins_df=length_bins(pep,args.bins,args.bootstraps,args.seed)
    quart_df=activity_groups(pep)
    outputs={
        'peptide_level_features.csv':pep,
        'sequence_qc_summary.csv':pd.DataFrame([qc]),
        'peptide_length_counts.csv':len_df,
        'residue_frequency.csv':fq,
        'length_activity_binned.csv':bins_df,
        'length_activity_association.csv':relation,
        'residue_activity_associations.csv':assoc,
        'activity_quartile_residue_composition.csv':quart_df,
    }
    for name, data in outputs.items():
        data.to_csv(args.output_dir/name,index=False,float_format='%.8g')
    paths=make_plot(pep,fq,bins_df,assoc,relation,args)
    print('Input:',args.input)
    print('Peptides:',len(pep),'(with activity:',int(pep.activity_is_finite.sum()),')')
    print('Lengths median [Q1,Q3]:',qc['median_length'], '[',qc['length_q1'],',',qc['length_q3'],']')
    print('Top five residues:')
    print(fq[['residue','pooled_frequency_percent','peptide_prevalence_percent']].head(5).to_string(index=False))
    print('Length–activity Spearman rho:',relation.iloc[0].spearman_rho_length_vs_endpoint)
    print('FDR-significant residue associations:',assoc.loc[assoc.fdr_significant_0_05,'residue'].tolist())
    print('Saved CSVs:')
    for file in outputs:
        print(' ',args.output_dir/file)
    print('Saved figures:')
    for file in paths:
        print(' ',file)


if __name__=='__main__':
    '''
    python analyze_ecoli_peptides.py \
    --input ./data/ecoli_five_fold_dataset.csv \
    --output-dir ./results/plots/pep_analysis \
    --activity-transform none \
    --duplicate-policy median \
    --bootstraps 300 \
    --seed 42 \
    --dpi 600 \
    --width 7.1 \
    --height 5.9
    
    '''
    
    
    main()
