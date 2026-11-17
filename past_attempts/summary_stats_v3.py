"""
Summary statistics analysis for v3 variability database.

This script loads the source_info_v3.pkl file and generates comprehensive
summary statistics on the variability data.
"""

import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from collections import OrderedDict


def load_v3_data(filepath='files/source_info_v3.pkl'):
    """
    Load the v3 variability database pickle file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the source_info_v3.pkl file
        
    Returns
    -------
    OrderedDict
        Dictionary containing variability info for all sources
    """
    info_file = Path(filepath)
    
    if not info_file.is_file():
        raise FileNotFoundError(f'Did not find {info_file}')
    
    with open(info_file, 'rb') as inp:
        sd = pickle.load(inp)
    
    print(f'Loaded wtlike-generated variability info for {len(sd)} sources')
    return sd


def extract_features(data):
    """
    Extract key features from the v3 database into a DataFrame.
    
    Parameters
    ----------
    data : OrderedDict
        Variability database from load_v3_data()
        
    Returns
    -------
    pd.DataFrame
        DataFrame with extracted features for each source
    """
    records = []
    
    for source_name, info in data.items():
        record = {'source': source_name}
        
        # Extract variability index if present
        if 'variability' in info:
            record['bbvar'] = info['variability']
        else:
            record['bbvar'] = np.nan
        
        # Extract light curve info
        lc = info.get('light_curve')
        if lc is not None:
            if isinstance(lc, dict):
                record['nblocks'] = len(lc.get('tw', []))
                if 'flux' in lc:
                    flux_values = lc['flux']
                    flux_vals = [
                        fv[0] if hasattr(fv, '__len__') and not isinstance(fv, (str, bytes)) else fv
                        for fv in flux_values
                    ]
                    flux_vals = np.array(flux_vals, dtype=float)
                    record['flux_mean'] = np.mean(flux_vals)
                    record['flux_std'] = np.std(flux_vals)
                    record['flux_min'] = np.min(flux_vals)
                    record['flux_max'] = np.max(flux_vals)
            else:
                record['nblocks'] = len(lc)
        else:
            record['nblocks'] = 0
        
        # Extract nearby source count
        nearby = info.get('nearby')
        record['nearby_count'] = len(nearby) if nearby is not None else 0
        
        # Extract FFT peaks if present
        fft_peaks = info.get('fft_peaks', [])
        record['nfft_peaks'] = len(fft_peaks)
        
        records.append(record)
    
    df = pd.DataFrame(records)
    return df


def compute_summary_stats(df):
    """
    Generate comprehensive summary statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from extract_features()
        
    Returns
    -------
    dict
        Dictionary containing various summary statistics
    """
    stats = {}
    
    # Overall counts
    stats['total_sources'] = len(df)
    stats['sources_with_lightcurves'] = (df['nblocks'] > 0).sum()
    stats['sources_with_variability'] = df['bbvar'].notna().sum()
    
    # Variability index statistics
    bbvar_valid = df['bbvar'].dropna()
    if len(bbvar_valid) > 0:
        stats['bbvar_mean'] = bbvar_valid.mean()
        stats['bbvar_std'] = bbvar_valid.std()
        stats['bbvar_median'] = bbvar_valid.median()
        stats['bbvar_min'] = bbvar_valid.min()
        stats['bbvar_max'] = bbvar_valid.max()
        stats['bbvar_q25'] = bbvar_valid.quantile(0.25)
        stats['bbvar_q75'] = bbvar_valid.quantile(0.75)
    
    # Light curve block statistics
    nblocks_valid = df['nblocks'][df['nblocks'] > 0]
    if len(nblocks_valid) > 0:
        stats['nblocks_mean'] = nblocks_valid.mean()
        stats['nblocks_median'] = nblocks_valid.median()
        stats['nblocks_max'] = nblocks_valid.max()
        stats['nblocks_min'] = nblocks_valid.min()
    
    # Flux statistics (if available)
    flux_mean_valid = df['flux_mean'].dropna()
    if len(flux_mean_valid) > 0:
        stats['flux_mean_overall'] = flux_mean_valid.mean()
        stats['flux_mean_median'] = flux_mean_valid.median()
    
    flux_std_valid = df['flux_std'].dropna()
    if len(flux_std_valid) > 0:
        stats['flux_variability_mean'] = flux_std_valid.mean()
    
    # Nearby sources statistics
    nearby_valid = df['nearby_count'][df['nearby_count'] > 0]
    if len(nearby_valid) > 0:
        stats['nearby_mean'] = nearby_valid.mean()
        stats['nearby_median'] = nearby_valid.median()
        stats['nearby_max'] = nearby_valid.max()
    
    # FFT peaks statistics
    fft_valid = df['nfft_peaks'][df['nfft_peaks'] > 0]
    if len(fft_valid) > 0:
        stats['fft_peaks_mean'] = fft_valid.mean()
        stats['fft_peaks_sources'] = len(fft_valid)
    
    return stats


def print_summary_report(stats, df):
    """
    Print a formatted summary report.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_summary_stats()
    df : pd.DataFrame
        Feature DataFrame from extract_features()
    """
    print("\n" + "="*70)
    print("SUMMARY STATISTICS FOR SOURCE_INFO_V3.pkl")
    print("="*70)
    
    print(f"\nSOURCE COUNTS:")
    print(f"  Total sources:                  {stats['total_sources']:>6}")
    print(f"  Sources with light curves:      {stats['sources_with_lightcurves']:>6}")
    print(f"  Sources with variability data:  {stats['sources_with_variability']:>6}")
    
    if 'bbvar_mean' in stats:
        print(f"\nVARIABILITY INDEX (bbvar) STATISTICS:")
        print(f"  Mean:                           {stats['bbvar_mean']:>10.2f}")
        print(f"  Median:                         {stats['bbvar_median']:>10.2f}")
        print(f"  Std Dev:                        {stats['bbvar_std']:>10.2f}")
        print(f"  Min:                            {stats['bbvar_min']:>10.2f}")
        print(f"  Max:                            {stats['bbvar_max']:>10.2f}")
        print(f"  Q1 (25%):                       {stats['bbvar_q25']:>10.2f}")
        print(f"  Q3 (75%):                       {stats['bbvar_q75']:>10.2f}")
    
    if 'nblocks_mean' in stats:
        print(f"\nLIGHT CURVE BLOCK STATISTICS:")
        print(f"  Mean blocks per curve:          {stats['nblocks_mean']:>10.2f}")
        print(f"  Median blocks:                  {stats['nblocks_median']:>10.2f}")
        print(f"  Max blocks:                     {stats['nblocks_max']:>10.0f}")
        print(f"  Min blocks:                     {stats['nblocks_min']:>10.0f}")
    
    if 'flux_mean_overall' in stats:
        print(f"\nFLUX STATISTICS:")
        print(f"  Mean flux (avg of sources):     {stats['flux_mean_overall']:>10.3e}")
        print(f"  Median flux:                    {stats['flux_mean_median']:>10.3e}")
        print(f"  Avg variability (std):          {stats['flux_variability_mean']:>10.3e}")
    
    if 'nearby_mean' in stats:
        print(f"\nNEARBY SOURCES STATISTICS:")
        print(f"  Mean nearby sources:            {stats['nearby_mean']:>10.2f}")
        print(f"  Median nearby:                  {stats['nearby_median']:>10.2f}")
        print(f"  Max nearby:                     {stats['nearby_max']:>10.0f}")
    
    if 'fft_peaks_sources' in stats:
        print(f"\nFFT PEAKS STATISTICS:")
        print(f"  Sources with FFT peaks:         {stats['fft_peaks_sources']:>6}")
        print(f"  Mean peaks per source:          {stats['fft_peaks_mean']:>10.2f}")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    # Load the v3 data
    v3_data = load_v3_data()
    
    # Extract features into a DataFrame
    df = extract_features(v3_data)
    
    # Compute summary statistics
    stats = compute_summary_stats(df)
    
    # Print formatted report
    print_summary_report(stats, df)
    
    # Display basic DataFrame info
    print("\nDATAFRAME OVERVIEW:")
    print(df.describe())
    
    # Save results to CSV
    output_csv = 'v3_summary_statistics.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nFull results saved to: {output_csv}")
    
    return df, stats


if __name__ == '__main__':
    df, stats = main()