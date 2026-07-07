"""
Enhanced functions for analyzing 3-block Bayesian Block light curves
Based on select_single_step from agn_steps.py, modified for 3-block analysis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utilities.ipynb_docgen import show
from pylib.data_setup import VarDB, set_theme


def select_double_steps(vdb, margin=50, bin_width=7):
    """
    Detect light curves with exactly three Bayesian blocks (two transitions/steps).
    
    This function identifies sources with two flux transitions and evaluates their
    variability. For 3-block light curves, we can characterize:
    - The first step ratio (block2/block1)
    - The second step ratio (block3/block2)
    - The overall variability across all three blocks
    - The time positions of both transitions
    
    Parameters
    ----------
    vdb : VarDB
        Variable database containing light curve information
    margin : int, optional
        Minimum width in weeks for edge blocks (default: 50)
    bin_width : int, optional
        Bin width in days used for light curve generation (default: 7)
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with the following columns:
        - flux_ratio_1: ratio of second to first block (b/a)
        - flux_ratio_2: ratio of third to second block (c/b)
        - flux_ratio_overall: ratio of third to first block (c/a)
        - flux_product: product of step ratios (b/a) * (c/b)
        - time_1: width of first block in weeks
        - time_2: width of second block in weeks
        - time_3: width of third block in weeks
        - ts: test statistic for variability
        - association: source type classification
        - log_eflux: log of 100 MeV flux
        - bbvar: Bayesian block variability index
        - variability: overall variability index
    """
    
    show(f"""### Detect sources with double-steps (three Bayesian blocks)
    
Here I select BB light curves with exactly three blocks, representing two distinct
transitions/steps in flux. For each source, I record:
1. The ratio of the two transitions (flux_ratio_1 and flux_ratio_2)
2. The overall step from first to third block (flux_ratio_overall)
3. The position and width of each block
4. Various variability measures

The light curves were generated with a bin width of {bin_width} days, so block transitions
must occur at bin boundaries. The BB algorithm determines the exact boundaries.

A source with three blocks indicates either:
- A transition from state A → B → C (two sequential steps)
- Statistical fluctuations in the data that create apparent blocks
- Complex variability with multiple timescales

Ratios close to 1.0 may indicate artifacts of the BB procedure; this needs study.
""")
    
    dfx = vdb.dfx
    ass = dfx.association.values
    tss = dfx.ts.values
    names = dfx.index
    
    def make_df(x):
        if x is None:
            return None
        return pd.DataFrame.from_dict(x)
    
    lcs = [make_df(vdb[uw_name]['light_curve']) for uw_name in dfx.uw_name]
    
    dd = dict()
    for name, lc, stype, ts in zip(names, lcs, ass, tss):
        # Only process light curves with exactly 3 blocks
        if lc is None or len(lc) != 3:
            continue
        
        v = lc.tw.values / 7  # Convert to weeks
        sh = lc.flux.values.shape
        
        # Handle different flux data shapes (new vs old format)
        if sh == (3,):
            a, b, c = lc.flux.values  # new format
        elif sh == (3, 2):
            a, b, c = lc.flux.values[:, 0]  # old format with two flux columns
        else:
            raise Exception(f'Bad lc.flux shape for 3 blocks: {sh}')
        
        # Quality check: all blocks should have positive flux
        # and edge blocks should meet the margin requirement
        if (a > 0) and (b > 0) and (c > 0) and (v[0] >= margin) and (v[-1] >= margin):
            
            # Calculate step ratios
            ratio_1 = b / a  # First transition
            ratio_2 = c / b  # Second transition
            ratio_overall = c / a  # Overall
            ratio_product = ratio_1 * ratio_2  # Should equal overall
            
            # Determine pattern type
            if ratio_1 > 1 and ratio_2 > 1:
                pattern = 'up-up'
            elif ratio_1 < 1 and ratio_2 < 1:
                pattern = 'down-down'
            elif ratio_1 > 1 and ratio_2 < 1:
                pattern = 'up-down'
            else:  # ratio_1 < 1 and ratio_2 > 1
                pattern = 'down-up'
            
            # Calculate variability metrics
            # Coefficient of variation (normalized to mean)
            mean_flux = np.mean([a, b, c])
            std_flux = np.std([a, b, c])
            cv = std_flux / mean_flux if mean_flux > 0 else 0
            
            dd[name] = dict(
                flux_ratio_1=ratio_1,
                flux_ratio_2=ratio_2,
                flux_ratio_overall=ratio_overall,
                flux_product=ratio_product,
                pattern=pattern,
                coeff_variation=cv,
                time_1=v[0],
                time_2=v[1],
                time_3=v[2],
                flux_a=a,
                flux_b=b,
                flux_c=c,
                ts=ts,
                association=stype
            )
    
    if not dd:
        raise ValueError('Failed to find any sources with exactly 3 Bayesian blocks!')
    
    df = pd.DataFrame.from_dict(dd, orient='index')
    
    # Add variability metrics from the full database
    df.loc[:, 'log_eflux'] = np.log10(dfx.loc[df.index, 'eflux100'])
    df.loc[:, 'bbvar'] = dfx.loc[df.index, 'bbvar']
    df.loc[:, 'variability'] = dfx.loc[dfx.index, 'variability']
    
    show(f"""Apply margin={margin} weeks: <br>Found {len(df)} candidates with 3 blocks, organized by association:""")
    assert len(df) > 0, 'Failed to find any 3-block sources!'
    v, n = np.unique(df.association, return_counts=True)
    show(pd.Series(dict(list(zip(v, n))), name='Count'))
    
    show(f"""Pattern distribution:""")
    pattern_counts = df.pattern.value_counts()
    show(pattern_counts)
    
    return df


def analyze_three_block_variability(df_three, margin, fignum=1):
    """
    Analyze and visualize variability of 3-block light curves.
    
    Creates diagnostic plots showing:
    - Distribution of first and second step ratios
    - Pattern types (up-up, down-down, up-down, down-up)
    - Relationship between steps and variability measures
    - Comparison across source types
    
    Parameters
    ----------
    df_three : pd.DataFrame
        DataFrame from select_double_steps
    margin : int
        Margin value used in selection (for documentation)
    fignum : int, optional
        Figure number for display (default: 1)
    """
    
    show(f"""## Analysis of Three-Block Light Curves
    
    These sources show two transitions in flux. Unlike 2-block sources which show
    a single sustained change, 3-block sources reveal more complex variability patterns.
    The steps may indicate:
    - Temporary states with limited duration
    - Multiple timescale behavior
    - Or statistical artifacts
    """)
    
    # Create a comprehensive figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Step ratio 1 vs Step ratio 2 (scatter)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(
        np.log10(df_three.flux_ratio_1),
        np.log10(df_three.flux_ratio_2),
        c=np.log10(df_three.ts),
        s=100,
        alpha=0.6,
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5
    )
    ax1.axhline(0, color='orange', ls='--', alpha=0.5)
    ax1.axvline(0, color='orange', ls='--', alpha=0.5)
    ax1.set_xlabel('Log₁₀(Step Ratio 1: b/a)')
    ax1.set_ylabel('Log₁₀(Step Ratio 2: c/b)')
    ax1.set_title('Two-Step Space')
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Log₁₀(TS)')
    
    # Plot 2: Pattern distribution (pie chart)
    ax2 = fig.add_subplot(gs[0, 1])
    pattern_counts = df_three.pattern.value_counts()
    colors = ['green', 'red', 'blue', 'orange']
    ax2.pie(
        pattern_counts.values,
        labels=pattern_counts.index,
        autopct='%1.1f%%',
        colors=colors[:len(pattern_counts)]
    )
    ax2.set_title('Pattern Distribution')
    
    # Plot 3: Coefficient of variation by pattern
    ax3 = fig.add_subplot(gs[0, 2])
    df_three.boxplot(column='coeff_variation', by='pattern', ax=ax3)
    ax3.set_xlabel('Pattern Type')
    ax3.set_ylabel('Coefficient of Variation')
    ax3.set_title('Variability by Pattern')
    plt.sca(ax3)
    plt.xticks(rotation=45)
    
    # Plot 4: First step ratio histogram
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(np.log10(df_three.flux_ratio_1), bins=30, alpha=0.7, color='cyan', edgecolor='black')
    ax4.axvline(0, color='red', ls='--', linewidth=2, label='No step')
    ax4.set_xlabel('Log₁₀(Step Ratio 1)')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of First Steps')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Plot 5: Second step ratio histogram
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(np.log10(df_three.flux_ratio_2), bins=30, alpha=0.7, color='magenta', edgecolor='black')
    ax5.axvline(0, color='red', ls='--', linewidth=2, label='No step')
    ax5.set_xlabel('Log₁₀(Step Ratio 2)')
    ax5.set_ylabel('Count')
    ax5.set_title('Distribution of Second Steps')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # Plot 6: Overall ratio histogram
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(np.log10(df_three.flux_ratio_overall), bins=30, alpha=0.7, color='yellow', edgecolor='black')
    ax6.axvline(0, color='red', ls='--', linewidth=2, label='No step')
    ax6.set_xlabel('Log₁₀(Overall Step Ratio)')
    ax6.set_ylabel('Count')
    ax6.set_title('Distribution of Overall Steps')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # Plot 7: Block widths
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(df_three.time_1, df_three.time_2, alpha=0.6, s=100, label='Block 1 vs 2')
    ax7.scatter(df_three.time_2, df_three.time_3, alpha=0.6, s=100, label='Block 2 vs 3')
    ax7.set_xlabel('Block Width (weeks)')
    ax7.set_ylabel('Block Width (weeks)')
    ax7.set_title('Block Width Relationships')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    # Plot 8: Variability by association
    ax8 = fig.add_subplot(gs[2, 1])
    associations = df_three.association.unique()
    for assoc in associations:
        subset = df_three[df_three.association == assoc]
        ax8.scatter(subset.index, subset.coeff_variation, label=assoc, alpha=0.6, s=80)
    ax8.set_ylabel('Coefficient of Variation')
    ax8.set_xlabel('Source Index')
    ax8.set_title('Variability by Source Type')
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3)
    
    # Plot 9: TS vs Coefficient of Variation
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.scatter(df_three.coeff_variation, df_three.ts, alpha=0.6, s=100, c='purple')
    ax9.set_xlabel('Coefficient of Variation')
    ax9.set_ylabel('Test Statistic (TS)')
    ax9.set_xscale('log')
    ax9.set_yscale('log')
    ax9.set_title('Variability Measure Correlation')
    ax9.grid(alpha=0.3, which='both')
    
    show(fig, fignum=fignum, caption="""
    Comprehensive analysis of three-block light curves:
    - Top row: Two-step space, pattern distribution, and variability by pattern
    - Middle row: Histograms of individual step ratios
    - Bottom row: Block width relationships, variability by source type, and correlation analysis
    """)


def three_block_statistics(df_three):
    """
    Generate detailed statistical summary for 3-block light curves.
    
    Parameters
    ----------
    df_three : pd.DataFrame
        DataFrame from select_double_steps
    """
    
    show(f"""## Statistical Summary of 3-Block Sources""")
    
    # Basic statistics
    show(f"""### Overview
    - Total sources with 3 blocks: {len(df_three)}
    - Median coefficient of variation: {df_three.coeff_variation.median():.3f}
    - Mean TS: {df_three.ts.mean():.1f}
    """)
    
    # Pattern analysis
    show(f"""### Pattern Analysis""")
    for pattern in df_three.pattern.unique():
        subset = df_three[df_three.pattern == pattern]
        show(f"""**{pattern.upper()}** ({len(subset)} sources):
        - Mean first step: {subset.flux_ratio_1.mean():.2f}x
        - Mean second step: {subset.flux_ratio_2.mean():.2f}x
        - Mean overall step: {subset.flux_ratio_overall.mean():.2f}x
        - Mean coefficient of variation: {subset.coeff_variation.mean():.3f}
        - Mean TS: {subset.ts.mean():.1f}
        """)
    
    # Association analysis
    show(f"""### By Source Type (Association)""")
    for assoc in df_three.association.unique():
        subset = df_three[df_three.association == assoc]
        show(f"""**{assoc}** ({len(subset)} sources):
        - Mean overall flux ratio: {subset.flux_ratio_overall.mean():.2f}x
        - Coefficient of variation range: [{subset.coeff_variation.min():.3f}, {subset.coeff_variation.max():.3f}]
        - Median TS: {subset.ts.median():.1f}
        """)
    
    # High variability sources
    show(f"""### Highest Variability Sources""")
    top_var = df_three.nlargest(5, 'coeff_variation')[['association', 'pattern', 'coeff_variation', 'ts']]
    show(top_var)


def compare_two_vs_three_blocks(df_two, df_three):
    """
    Compare properties of 2-block and 3-block light curves.
    
    Parameters
    ----------
    df_two : pd.DataFrame
        DataFrame from select_single_step (2-block sources)
    df_three : pd.DataFrame
        DataFrame from select_double_steps (3-block sources)
    """
    
    show(f"""## Comparison: 2-Block vs 3-Block Light Curves
    
    Understanding the relationship between single-step (2-block) and double-step (3-block)
    sources helps distinguish between:
    - True astrophysical variability (should show coherent behavior)
    - BB algorithm artifacts (should be independent or random)
    """)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: TS distribution
    axes[0, 0].hist(df_two.ts, bins=30, alpha=0.6, label=f'2-block (n={len(df_two)})', color='blue', edgecolor='black')
    axes[0, 0].hist(df_three.ts, bins=30, alpha=0.6, label=f'3-block (n={len(df_three)})', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Test Statistic (TS)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('TS Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Source type distribution
    two_assoc = df_two.association.value_counts()
    three_assoc = df_three.association.value_counts()
    x = np.arange(len(two_assoc))
    width = 0.35
    axes[0, 1].bar(x - width/2, two_assoc.values, width, label='2-block', alpha=0.7)
    axes[0, 1].bar(x + width/2, three_assoc.values, width, label='3-block', alpha=0.7)
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Association Type Distribution')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(two_assoc.index, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    # Plot 3: BBvar distribution
    axes[1, 0].hist(df_two.bbvar, bins=30, alpha=0.6, label='2-block', color='blue', edgecolor='black')
    axes[1, 0].hist(df_three.bbvar, bins=30, alpha=0.6, label='3-block', color='red', edgecolor='black')
    axes[1, 0].set_xlabel('BBvar Index')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('BBvar Distribution')
    axes[1, 0].set_xscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Log eflux distribution
    axes[1, 1].scatter(df_two.log_eflux, df_two.ts, alpha=0.6, s=50, label='2-block')
    axes[1, 1].scatter(df_three.log_eflux, df_three.ts, alpha=0.6, s=50, label='3-block')
    axes[1, 1].set_xlabel('Log₁₀(eflux100)')
    axes[1, 1].set_ylabel('Test Statistic (TS)')
    axes[1, 1].set_title('Flux vs TS')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    show(fig, caption="""Comparison between 2-block (single-step) and 3-block (double-step) sources.
    These comparisons help identify whether 3-block sources are astrophysically distinct or
    statistical artifacts of the Bayesian blocks algorithm.""")