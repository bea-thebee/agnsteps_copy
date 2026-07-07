"""
Monte Carlo Simulation Example using AGNstepsMC
Based on parameters from mc_dev.py in tburnett/agnsteps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylib.mc_dev import AGNstepsMC, BBsim_plots
from utilities.ipynb_docgen import show, capture
from wtlike import MJD, UTC

# ============================================================================
# Setup Parameters (from mc_dev.py)
# ============================================================================

# AGNstepsMC initialization parameters
SOURCE_NAME = "example_source"    # Your AGN source name
BINSIZE = 30                       # bin size in days
DELTA = 900                        # time window around step (±delta)

# BBsim.run_many parameters
N_SIMULATIONS = 1000              # number of simulations to run
NPROC = 10                         # number of processors for parallelization
SEEDS = range(N_SIMULATIONS)       # random seeds for reproducibility

# ============================================================================
# Main Simulation Workflow
# ============================================================================

def run_simulation():
    """
    Execute a complete Monte Carlo simulation following the AGNstepsMC pattern.
    """
    
    print("=" * 70)
    print("AGNsteps Monte Carlo Simulation")
    print("=" * 70)
    
    # Step 1: Initialize the Monte Carlo setup
    print("\n[1] Initializing AGNstepsMC with parameters:")
    print(f"    - Source: {SOURCE_NAME}")
    print(f"    - Bin size: {BINSIZE} days")
    print(f"    - Time window: ±{DELTA} (around detected step)")
    
    agn_mc = AGNstepsMC(
        name=SOURCE_NAME,
        binsize=BINSIZE,
        delta=DELTA
    )
    
    # Step 2: Display light curve plots
    print("\n[2] Generating light curve plots...")
    agn_mc.lc_plots()
    
    # Step 3: Run single simulation (test)
    print("\n[3] Running a single test simulation...")
    single_result = agn_mc.simulate(step_time=None)
    print(f"    Result: {single_result}")
    
    # Step 4: Run multiple simulations
    print(f"\n[4] Running {N_SIMULATIONS} simulations on {NPROC} processors...")
    sim_results, df_results = agn_mc.multi_simulate(N=N_SIMULATIONS, nproc=NPROC)
    
    print(f"    Simulation complete!")
    print(f"    Results shape: {df_results.shape}")
    print(f"\n    Results summary:")
    print(df_results.describe())
    
    # Step 5: Generate plots
    print("\n[5] Generating result plots...")
    interval = agn_mc.partition_view.cells.tw.values.mean()
    plots = BBsim_plots(
        sim=sim_results,
        df_all=df_results,
        step=agn_mc.step,
        interval=interval
    )
    
    # Plot histograms for different block configurations
    print("\n    Plotting results for 2-block configurations (nbb==2)...")
    plots.hists(cut='nbb==2', dtmax=40)
    
    # Step 6: Save results
    print("\n[6] Saving results to CSV...")
    output_file = f"simulation_results_{SOURCE_NAME}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"    Saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)
    
    return agn_mc, sim_results, df_results


if __name__ == "__main__":
    # Run the simulation
    agn_mc, sim_results, df_results = run_simulation()
    
    # Example: Access specific results
    print("\n" + "=" * 70)
    print("Detailed Results Analysis")
    print("=" * 70)
    
    print(f"\nDetected Step Information:")
    print(f"  - Step time (MJD): {agn_mc.step.time:.2f}")
    print(f"  - Variability TS: {agn_mc.step.ts:.1f}")
    print(f"  - Flux ratio (after/before): {agn_mc.step.ratio:.2f}")
    print(f"  - Number of blocks: {agn_mc.step.nbb}")
    
    # Filter results
    two_block_results = df_results[df_results['nbb'] == 2]
    print(f"\nResults with 2 blocks (clear steps):")
    print(f"  - Count: {len(two_block_results)}")
    print(f"  - Mean TS: {two_block_results['ts'].mean():.1f}")
    print(f"  - Mean time offset: {(two_block_results['time'] - agn_mc.step.time).mean():.1f} MJD")
    print(f"  - Mean flux ratio: {two_block_results['ratio'].mean():.2f}")






"""
Enhanced flux_figure method for J2333_plots
Compares average flux measurements from Fermi data across specified time periods
"""



def flux_figure_from_averages(self, fermi_periods=None, reference_flux=5.9, 
                              title="Compare Fermi Flux Measurements"):
    """
    Compare average flux measurements from Fermi data with BB light curve.
    
    Instead of using hardcoded dates/fluxes, this method computes average
    fluxes from the BB light curve for specified time periods.
    
    Parameters
    ----------
    fermi_periods : list of tuples, optional
        List of (start_date, end_date) tuples defining Fermi observation periods.
        Dates can be strings ('YYYY-MM-DD'), MJD floats, or UTC datetime objects.
        If None, will use periods from self.bb.fluxes.
        Example: [('2015-01-01', '2016-01-01'), ('2018-01-05', '2019-09-30')]
    
    reference_flux : float, optional
        Reference flux for normalization (default: 5.9, the pointlike average).
        Fluxes will be divided by this value for relative comparison.
    
    title : str, optional
        Title for the plot description.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    fermi_fluxes : pd.DataFrame
        DataFrame with columns: 'start', 'end', 'mjd_center', 'width', 
        'avg_flux', 'flux_error', 'relative_flux'
    """
    
    show(f"""## {title}""")
    
    # Get the BB light curve data
    df_bb = self.bb.fluxes
    
    # If no periods specified, use the BB blocks as periods
    if fermi_periods is None:
        show("No periods specified. Using BB light curve blocks as Fermi periods.")
        # Create periods from BB blocks
        fermi_periods = []
        for i, row in df_bb.iterrows():
            start = row['t'] - row['tw'] / 2
            end = row['t'] + row['tw'] / 2
            fermi_periods.append((start, end))
    else:
        # Convert string dates to MJD if needed
        converted_periods = []
        for start, end in fermi_periods:
            if isinstance(start, str):
                start = MJD(start)
            if isinstance(end, str):
                end = MJD(end)
            converted_periods.append((start, end))
        fermi_periods = converted_periods
    
    # Calculate average fluxes for each Fermi period
    fermi_fluxes_list = []
    for i, (tstart, tstop) in enumerate(fermi_periods):
        # Find BB blocks that overlap with this period
        mask = (df_bb['t'] >= tstart) & (df_bb['t'] <= tstop)
        
        if mask.sum() > 0:
            # Get fluxes in this period
            period_fluxes = df_bb.loc[mask, 'flux'].values
            period_errors = df_bb.loc[mask, 'errors'].values
            
            # Calculate weighted average (inverse variance weighting)
            weights = 1.0 / (period_errors ** 2)
            avg_flux = np.average(period_fluxes, weights=weights)
            flux_error = np.sqrt(1.0 / weights.sum())
            
            mjd_center = (tstart + tstop) / 2
            width = (tstop - tstart) / 2
            relative_flux = avg_flux / reference_flux
            relative_error = flux_error / reference_flux
            
            fermi_fluxes_list.append({
                'start': tstart,
                'end': tstop,
                'mjd_center': mjd_center,
                'width': width,
                'avg_flux': avg_flux,
                'flux_error': flux_error,
                'relative_flux': relative_flux,
                'relative_error': relative_error,
            })
        else:
            show(f"Warning: No BB blocks found in period {i} ({UTC(tstart)} to {UTC(tstop)})")
    
    fermi_df = pd.DataFrame(fermi_fluxes_list)
    
    # Create the comparison figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot Fermi average measurements with error bars
    ax.errorbar(
        x=fermi_df['mjd_center'].values,
        xerr=fermi_df['width'].values,
        y=fermi_df['relative_flux'].values,
        yerr=fermi_df['relative_error'].values,
        fmt=' ',
        marker='o',
        ms=12,
        color='magenta',
        label='Fermi Average (calculated)',
        zorder=5
    )
    
    # Plot the full BB light curve
    ax.set(xticks=np.arange(56000, 60001, 1000))
    fig = self.bb.plot(
        ax=ax,
        ylim=(0, 2.4),
        colors=('none', 'none', 'maroon'),
        legend_loc='upper left',
        source_name=''
    )
    
    show(fig)
    
    # Display results table
    show(fermi_df[['start', 'end', 'relative_flux', 'relative_error']], 
         summary='Fermi Flux Averages (Relative to Pointlike)')
    
    # Generate summary notes
    mjd_transition = df_bb.t.iloc[0] + df_bb.tw.iloc[0] / 2
    
    notes = f"""## Analysis Notes:

* **Reference Flux**: {reference_flux}×10⁻¹² erg cm⁻² s⁻¹ (pointlike average)
* **Fluxes Displayed**: Relative to reference (divided by {reference_flux})
* **Transition Time**: MJD {mjd_transition:.1f} ({UTC(mjd_transition)[:-6]}), within detection resolution

### Fermi Period Summary:
"""
    for i, row in fermi_df.iterrows():
        period_label = f"Period {i+1}"
        date_start = UTC(row['start'])[:-6]
        date_end = UTC(row['end'])[:-6]
        notes += f"\n* **{period_label}**: {date_start} to {date_end}"
        notes += f"\n  - Relative Flux: {row['relative_flux']:.2f} ± {row['relative_error']:.3f}"
        notes += f"\n  - Absolute Flux: {row['avg_flux']:.2e}"
    
    show(notes)
    
    return fig, fermi_df


def flux_figure_custom_dates(self, fermi_dates_fluxes, reference_flux=5.9,
                            title="Compare Fermi Flux Measurements"):
    """
    Compare hardcoded Fermi measurements with BB light curve (original style).
    
    This is a wrapper to maintain backward compatibility with the original
    flux_figure method that uses hardcoded dates and flux values.
    
    Parameters
    ----------
    fermi_dates_fluxes : list of dicts
        List of measurement dictionaries with keys:
        - 'start': start date (string or MJD)
        - 'end': end date (string or MJD)
        - 'flux': measured flux value
        - 'label': optional label for the measurement (default: auto-generated)
    
    reference_flux : float, optional
        Reference flux for normalization.
    
    title : str, optional
        Title for the plot description.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    
    show(f"""## {title}""")
    
    # Convert to required format for flux_figure_from_averages
    periods = []
    fluxes_normalized = []
    labels = []
    
    for i, entry in enumerate(fermi_dates_fluxes):
        start = entry['start'] if isinstance(entry['start'], (int, float)) else MJD(entry['start'])
        end = entry['end'] if isinstance(entry['end'], (int, float)) else MJD(entry['end'])
        flux = entry['flux']
        label = entry.get('label', f"Measurement {i+1}")
        
        periods.append((start, end))
        fluxes_normalized.append(flux / reference_flux)
        labels.append(label)
    
    # Create the comparison figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot provided measurements
    px = np.array([0.5 * (p[0] + p[1]) for p in periods])
    pxerr = np.array([(p[1] - p[0]) / 2 for p in periods])
    
    ax.errorbar(
        x=px,
        xerr=pxerr,
        y=fluxes_normalized,
        fmt=' ',
        marker='o',
        ms=10,
        color='magenta',
        label='Fermi Measurements'
    )
    
    ax.set(xticks=np.arange(56000, 60001, 1000))
    
    # Plot the full BB light curve
    fig = self.bb.plot(
        ax=ax,
        ylim=(0, 2.4),
        colors=('none', 'none', 'maroon'),
        legend_loc='upper left',
        source_name=''
    )
    
    show(fig)
    
    # Display BB flux table
    df = self.bb.fluxes
    show(df, summary='BB Flux Table')
    
    return fig