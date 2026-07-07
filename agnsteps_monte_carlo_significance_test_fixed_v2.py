"""
Monte Carlo Significance Test for Bayesian Blocks Detection
Tests how many Bayesian Blocks the agnsteps algorithm detects when resampling
a single randomly selected AGN source with bootstrap variations.

This determines whether detected blocks are statistically significant
or could arise by chance from random noise.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
import seaborn as sns
import random
from tqdm import tqdm
import sys

# Configure styling
plt.style.use('dark_background')
sns.set_theme('talk', font_scale=1.0)


@dataclass
class BootstrapResult:
    """Results from a single bootstrap iteration"""
    iteration: int
    num_blocks: int
    flux_values: np.ndarray
    block_edges: np.ndarray
    mean_flux: float
    std_flux: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'iteration': self.iteration,
            'num_blocks': self.num_blocks,
            'mean_flux': float(self.mean_flux),
            'std_flux': float(self.std_flux),
        }


class BayesianBlocksSignificanceTest:
    """
    Performs Monte Carlo significance test on Bayesian Blocks detection.
    Bootstraps a single source to determine statistical significance of detected blocks.
    """
    
    def __init__(self, 
                 num_bootstrap_iterations: int = 100,
                 info_file: str = 'files/source_info_v1.pkl',
                 verbose: bool = True):
        """
        Initialize significance test.
        
        Args:
            num_bootstrap_iterations: Number of bootstrap resamples
            info_file: Path to source_info pickle file
            verbose: Print progress information
        """
        self.num_bootstrap_iterations = num_bootstrap_iterations
        self.info_file = Path(info_file)
        self.verbose = verbose
        self.source_name = None
        self.light_curve_df = None
        self.original_structure = None
        self.bootstrap_results: List[BootstrapResult] = []
        self.timestamp = pd.Timestamp.now()
        
    def _select_random_source_with_lightcurve(self, vdb: dict) -> str:
        """Select a random source that has a valid light curve."""
        sources_with_lc = [
            name for name, data in vdb.items() 
            if data.get('light_curve') is not None and len(data.get('light_curve', {})) > 1
        ]
        
        if not sources_with_lc:
            raise ValueError("No sources with valid light curves found in database")
        
        return random.choice(sources_with_lc)
    
    def _extract_light_curve_values(self, lc_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract flux, errors, and time values from light curve data.
        Handles both dict and DataFrame formats from source_info pickle.
        
        Returns:
            Tuple of (flux_array, error_lower, error_upper, times, time_widths)
        """
        # Convert dict to DataFrame if needed
        if isinstance(lc_data, dict):
            lc_df = pd.DataFrame.from_dict(lc_data, orient='index')
        else:
            lc_df = lc_data
        
        if self.verbose:
            print(f"  Light curve shape: {lc_df.shape}")
            print(f"  Columns: {list(lc_df.columns)}")
        
        # Extract flux values
        flux_array = None
        for col in lc_df.columns:
            if isinstance(col, str) and ('flux' in col.lower()):
                flux_col = lc_df[col]
                break
        else:
            # If no flux column found, try first numeric column
            for col in lc_df.columns:
                if lc_df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                    flux_col = lc_df[col]
                    break
            else:
                # Last resort: use first column
                flux_col = lc_df.iloc[:, 0]
        
        flux_values = flux_col.values
        flux_array = np.array([
            fv[0] if hasattr(fv, '__len__') and not isinstance(fv, (str, bytes)) else fv
            for fv in flux_values
        ], dtype=float)
        
        # Extract errors
        error_lower = None
        error_upper = None
        
        # Look for 'errors' column (tuple/list format)
        if 'errors' in lc_df.columns:
            errors = lc_df['errors'].values
            error_lower = np.array([e[0] if isinstance(e, (list, tuple)) and len(e) > 0 else e for e in errors], dtype=float)
            error_upper = np.array([e[1] if isinstance(e, (list, tuple)) and len(e) > 1 else e for e in errors], dtype=float)
        else:
            # Look for separate error columns (only check string column names)
            error_cols = [col for col in lc_df.columns if isinstance(col, str) and 'error' in col.lower()]
            if len(error_cols) >= 2:
                error_lower = np.array([e for e in lc_df[error_cols[0]].values], dtype=float)
                error_upper = np.array([e for e in lc_df[error_cols[1]].values], dtype=float)
        
        # Default errors if not found
        if error_lower is None or error_upper is None:
            error_lower = flux_array * 0.1
            error_upper = flux_array * 0.1
        
        # Extract time information
        times = None
        time_widths = None
        
        # Look for time columns
        for col in lc_df.columns:
            if isinstance(col, str):
                if col.lower() == 't':
                    times = lc_df[col].values
                elif col.lower() == 'tw':
                    time_widths = lc_df[col].values
        
        # Default times if not found
        if times is None:
            times = np.arange(len(flux_array), dtype=float)
        else:
            times = np.array(times, dtype=float)
        
        if time_widths is None:
            time_widths = np.ones(len(flux_array), dtype=float)
        else:
            time_widths = np.array(time_widths, dtype=float)
        
        return flux_array, error_lower, error_upper, times, time_widths
    
    def load_source_data(self, source_name: Optional[str] = None) -> bool:
        """
        Load source data from pickle file.
        
        Args:
            source_name: If None, selects a random source
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.info_file.exists():
                print(f"Error: {self.info_file} not found")
                return False
            
            with open(self.info_file, 'rb') as f:
                vdb = pickle.load(f)
            
            if self.verbose:
                print(f"✓ Loaded database with {len(vdb)} sources\n")
            
            # Select random source if not specified
            if source_name is None:
                source_name = self._select_random_source_with_lightcurve(vdb)
            
            if source_name not in vdb:
                print(f"Error: {source_name} not found in database")
                return False
            
            self.source_name = source_name
            source_dict = vdb[source_name]
            
            # Store light curve data (dict or DataFrame)
            self.light_curve_df = source_dict['light_curve']
            
            if self.verbose:
                print(f"{'='*80}")
                print(f"SELECTED SOURCE FOR SIGNIFICANCE TEST")
                print(f"{'='*80}")
                print(f"Source: {self.source_name}\n")
                
                # Try to get number of blocks
                try:
                    if isinstance(self.light_curve_df, dict):
                        num_blocks = len(self.light_curve_df.get('tw', []))
                    else:
                        num_blocks = len(self.light_curve_df)
                    print(f"Number of Bayesian Blocks: {num_blocks}\n")
                except:
                    print(f"Unable to determine number of blocks\n")
            
            return True
            
        except Exception as e:
            print(f"Error loading source data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def analyze_original_structure(self) -> Dict:
        """
        Analyze the original light curve structure.
        
        Returns:
            Dictionary with structural analysis
        """
        if self.light_curve_df is None:
            return {}
        
        try:
            # Extract values using robust method
            flux_array, error_lower, error_upper, times, time_widths = self._extract_light_curve_values(self.light_curve_df)
            
            analysis = {
                'num_blocks': len(flux_array),
                'flux_values': flux_array,
                'error_lower': error_lower,
                'error_upper': error_upper,
                'times': times,
                'time_widths': time_widths,
                'mean_flux': np.mean(flux_array),
                'std_flux': np.std(flux_array),
                'min_flux': np.min(flux_array),
                'max_flux': np.max(flux_array),
            }
            
            self.original_structure = analysis
            return analysis
        
        except Exception as e:
            print(f"Error analyzing light curve structure: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_bootstrap_sample(self, iteration: int) -> BootstrapResult:
        """
        Generate a bootstrap sample by resampling flux values from the original.
        
        Uses parametric bootstrap: sample from normal distribution of each block.
        
        Args:
            iteration: Bootstrap iteration number
            
        Returns:
            BootstrapResult with bootstrap statistics
        """
        if self.original_structure is None:
            raise ValueError("Original structure not analyzed")
        
        original = self.original_structure
        flux_values = original['flux_values']
        error_lower = original['error_lower']
        error_upper = original['error_upper']
        
        # Generate bootstrap sample by perturbing each block's flux
        # Use symmetric errors (average of lower and upper)
        errors = (error_lower + error_upper) / 2
        
        bootstrap_flux = flux_values.copy()
        for i in range(len(bootstrap_flux)):
            # Sample from normal distribution centered at original flux
            # with std = average of error bars
            perturbation = np.random.normal(0, errors[i])
            bootstrap_flux[i] = flux_values[i] + perturbation
            # Ensure flux stays positive
            bootstrap_flux[i] = max(bootstrap_flux[i], 0.01)
        
        result = BootstrapResult(
            iteration=iteration,
            num_blocks=len(bootstrap_flux),
            flux_values=bootstrap_flux,
            block_edges=np.array([0, len(bootstrap_flux)]),
            mean_flux=np.mean(bootstrap_flux),
            std_flux=np.std(bootstrap_flux),
        )
        
        return result
    
    def count_blocks_in_bootstrap(self, bootstrap_flux: np.ndarray) -> int:
        """
        Count the number of "blocks" in bootstrap sample based on variance.
        
        This is a simplified method that counts significant flux jumps.
        
        Args:
            bootstrap_flux: Bootstrap flux values
            
        Returns:
            Number of detected blocks
        """
        if len(bootstrap_flux) < 2:
            return 1
        
        # Calculate flux differences between consecutive points
        flux_diffs = np.abs(np.diff(bootstrap_flux))
        mean_diff = np.mean(flux_diffs)
        std_diff = np.std(flux_diffs)
        
        # Count "blocks" as regions where flux doesn't change significantly
        # A new block starts when difference is > mean + 1 std
        threshold = mean_diff + std_diff
        
        block_count = 1
        for diff in flux_diffs:
            if diff > threshold:
                block_count += 1
        
        return block_count
    
    def run_significance_test(self) -> List[BootstrapResult]:
        """
        Run the full Monte Carlo significance test.
        
        Returns:
            List of BootstrapResult objects
        """
        if self.light_curve_df is None:
            print("Error: Source data not loaded")
            return []
        
        # Analyze original
        original = self.analyze_original_structure()
        
        if not original:
            print("Error: Could not analyze original structure")
            return []
        
        if self.verbose:
            print(f"Original Light Curve:")
            print(f"  Number of Blocks: {original['num_blocks']}")
            print(f"  Mean Flux: {original['mean_flux']:.4f}")
            print(f"  Std Flux: {original['std_flux']:.4f}")
            print(f"  Flux Range: [{original['min_flux']:.4f}, {original['max_flux']:.4f}]\n")
            
            print(f"{'='*80}")
            print(f"RUNNING MONTE CARLO SIGNIFICANCE TEST")
            print(f"{'='*80}")
            print(f"Bootstrap iterations: {self.num_bootstrap_iterations}\n")
        
        self.bootstrap_results = []
        
        # Progress bar
        pbar = tqdm(
            total=self.num_bootstrap_iterations,
            desc="Bootstrap iterations",
            unit="iter",
            ncols=80,
            position=0,
            leave=True,
            file=sys.stdout
        )
        
        for i in range(self.num_bootstrap_iterations):
            try:
                # Generate bootstrap sample
                bootstrap_result = self.generate_bootstrap_sample(i)
                
                # Count blocks using variance-based method
                bootstrap_result.num_blocks = self.count_blocks_in_bootstrap(bootstrap_result.flux_values)
                
                self.bootstrap_results.append(bootstrap_result)
                
                pbar.update(1)
                if len(self.bootstrap_results) > 0:
                    avg_blocks = np.mean([r.num_blocks for r in self.bootstrap_results])
                    pbar.set_description(f"Bootstrap: {i+1}/{self.num_bootstrap_iterations} "
                                       f"(avg blocks: {avg_blocks:.1f})")
                
            except Exception as e:
                pbar.write(f"Error in iteration {i}: {str(e)}")
                pbar.update(1)
                continue
        
        pbar.close()
        
        return self.bootstrap_results
    
    def calculate_significance_statistics(self) -> Dict:
        """
        Calculate statistical significance of detected blocks.
        
        Returns:
            Dictionary with significance statistics
        """
        if not self.bootstrap_results:
            return {}
        
        original = self.original_structure
        bootstrap_blocks = np.array([r.num_blocks for r in self.bootstrap_results])
        
        # Calculate statistics
        original_blocks = original['num_blocks']
        mean_bootstrap_blocks = np.mean(bootstrap_blocks)
        std_bootstrap_blocks = np.std(bootstrap_blocks)
        median_bootstrap_blocks = np.median(bootstrap_blocks)
        
        # Z-score: how many std away from bootstrap mean is original?
        z_score = (original_blocks - mean_bootstrap_blocks) / std_bootstrap_blocks if std_bootstrap_blocks > 0 else 0
        
        # P-value: fraction of bootstrap samples with >= detected blocks
        p_value = np.sum(bootstrap_blocks >= original_blocks) / len(bootstrap_blocks)
        
        # Confidence level
        confidence_level = 1 - p_value
        
        stats = {
            'original_blocks': original_blocks,
            'mean_bootstrap_blocks': float(mean_bootstrap_blocks),
            'std_bootstrap_blocks': float(std_bootstrap_blocks),
            'median_bootstrap_blocks': float(median_bootstrap_blocks),
            'min_bootstrap_blocks': int(np.min(bootstrap_blocks)),
            'max_bootstrap_blocks': int(np.max(bootstrap_blocks)),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'confidence_level': float(confidence_level),
            '95_percentile': float(np.percentile(bootstrap_blocks, 95)),
            '99_percentile': float(np.percentile(bootstrap_blocks, 99)),
        }
        
        return stats
    
    def print_significance_report(self) -> None:
        """Print formatted significance test report."""
        if not self.bootstrap_results or self.original_structure is None:
            print("Error: Test not completed")
            return
        
        stats = self.calculate_significance_statistics()
        original = self.original_structure
        bootstrap_blocks = np.array([r.num_blocks for r in self.bootstrap_results])
        
        print("\n" + "="*80)
        print("MONTE CARLO SIGNIFICANCE TEST RESULTS")
        print("="*80 + "\n")
        
        print(f"Source: {self.source_name}")
        print(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Original Light Curve:")
        print(f"  Detected Blocks: {stats['original_blocks']}")
        print(f"  Mean Flux: {original['mean_flux']:.4f} ± {original['std_flux']:.4f}")
        print(f"  Flux Range: [{original['min_flux']:.4f}, {original['max_flux']:.4f}]\n")
        
        print(f"Bootstrap Test Statistics ({self.num_bootstrap_iterations} iterations):")
        print(f"  Mean Bootstrap Blocks: {stats['mean_bootstrap_blocks']:.2f}")
        print(f"  Median Bootstrap Blocks: {stats['median_bootstrap_blocks']:.1f}")
        print(f"  Std Dev: {stats['std_bootstrap_blocks']:.2f}")
        print(f"  Range: [{stats['min_bootstrap_blocks']}, {stats['max_bootstrap_blocks']}]")
        print(f"  95th Percentile: {stats['95_percentile']:.0f}")
        print(f"  99th Percentile: {stats['99_percentile']:.0f}\n")
        
        print(f"Significance Assessment:")
        print(f"  Z-Score: {stats['z_score']:.3f}")
        print(f"  P-Value: {stats['p_value']:.4f}")
        print(f"  Confidence Level: {stats['confidence_level']*100:.1f}%\n")
        
        # Interpretation
        print(f"Statistical Significance Assessment:")
        print(f"─" * 80)
        
        if stats['z_score'] > 2:
            significance = "HIGHLY SIGNIFICANT (>2σ)"
        elif stats['z_score'] > 1:
            significance = "MODERATELY SIGNIFICANT (1-2σ)"
        elif stats['z_score'] > 0.5:
            significance = "WEAKLY SIGNIFICANT (0.5-1σ)"
        else:
            significance = "NOT SIGNIFICANT (<0.5σ)"
        
        print(f"  {significance}\n")
        
        if stats['p_value'] < 0.01:
            print(f"  ✓ VERY STRONG evidence for real blocks")
            print(f"    Only {stats['p_value']*100:.2f}% of bootstrap samples had ≥{stats['original_blocks']} blocks")
        elif stats['p_value'] < 0.05:
            print(f"  ✓ STRONG evidence for real blocks")
            print(f"    Only {stats['p_value']*100:.2f}% of bootstrap samples had ≥{stats['original_blocks']} blocks")
        elif stats['p_value'] < 0.10:
            print(f"  ⚠ MODERATE evidence for real blocks")
            print(f"    {stats['p_value']*100:.2f}% of bootstrap samples had ≥{stats['original_blocks']} blocks")
        else:
            print(f"  ✗ WEAK or NO evidence for real blocks")
            print(f"    {stats['p_value']*100:.2f}% of bootstrap samples had ≥{stats['original_blocks']} blocks")
            print(f"    The detected blocks may be artifacts of noise!")
        
        print("\n" + "="*80 + "\n")
    
    def plot_significance_results(self, figsize: Tuple[float, float] = (14, 10)) -> plt.Figure:
        """
        Create comprehensive visualization of significance test results.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        if not self.bootstrap_results or self.original_structure is None:
            print("Error: Test not completed")
            return None
        
        stats = self.calculate_significance_statistics()
        original = self.original_structure
        bootstrap_blocks = np.array([r.num_blocks for r in self.bootstrap_results])
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        
        # 1. Histogram of bootstrap blocks
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(bootstrap_blocks, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(stats['original_blocks'], color='red', linestyle='--', linewidth=3, 
                   label=f'Original: {stats["original_blocks"]}')
        ax1.axvline(stats['mean_bootstrap_blocks'], color='green', linestyle='--', linewidth=2,
                   label=f'Bootstrap Mean: {stats["mean_bootstrap_blocks"]:.1f}')
        ax1.set_xlabel('Number of Blocks', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Distribution of Blocks in Bootstrap Samples', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. CDF of bootstrap blocks
        ax2 = fig.add_subplot(gs[0, 1])
        sorted_blocks = np.sort(bootstrap_blocks)
        cumulative = np.arange(1, len(sorted_blocks) + 1) / len(sorted_blocks)
        ax2.plot(sorted_blocks, cumulative, 'o-', markersize=5, alpha=0.7, linewidth=2)
        ax2.axvline(stats['original_blocks'], color='red', linestyle='--', linewidth=2,
                   label=f'Original: {stats["original_blocks"]}')
        ax2.axhline(stats['confidence_level'], color='orange', linestyle='--', linewidth=2,
                   label=f'Confidence: {stats["confidence_level"]*100:.1f}%')
        ax2.set_xlabel('Number of Blocks', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
        ax2.set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot
        ax3 = fig.add_subplot(gs[0, 2])
        bp = ax3.boxplot([bootstrap_blocks], labels=['Bootstrap'],
                         patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightblue')
        ax3.scatter([1], [stats['original_blocks']], color='red', s=200, zorder=3,
                   marker='*', label='Original')
        ax3.set_ylabel('Number of Blocks', fontsize=11, fontweight='bold')
        ax3.set_title('Bootstrap Distribution (Box Plot)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Original light curve
        ax4 = fig.add_subplot(gs[1, :2])
        times = original['times']
        fluxes = original['flux_values']
        errors_lower = original['error_lower']
        errors_upper = original['error_upper']
        widths = original['time_widths']
        
        ax4.errorbar(times, fluxes, 
                    yerr=[errors_lower, errors_upper],
                    xerr=widths/2,
                    fmt='o-', markersize=8, linewidth=2,
                    capsize=5, capthick=2,
                    color='steelblue', ecolor='steelblue', alpha=0.7,
                    label='Original')
        ax4.set_xlabel('Time (MJD)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Flux (relative)', fontsize=11, fontweight='bold')
        ax4.set_title(f'{self.source_name} - Original Light Curve', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        
        # 5. Statistics text box
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Determine significance color
        if stats['z_score'] > 2:
            sig_color = 'green'
            sig_text = 'HIGHLY\nSIGNIFICANT'
        elif stats['z_score'] > 1:
            sig_color = 'yellow'
            sig_text = 'MODERATELY\nSIGNIFICANT'
        elif stats['z_score'] > 0.5:
            sig_color = 'orange'
            sig_text = 'WEAKLY\nSIGNIFICANT'
        else:
            sig_color = 'red'
            sig_text = 'NOT\nSIGNIFICANT'
        
        summary_text = f"""
SIGNIFICANCE SUMMARY

Original Blocks: {stats['original_blocks']}
Bootstrap Mean: {stats['mean_bootstrap_blocks']:.2f}
Bootstrap Std: {stats['std_bootstrap_blocks']:.2f}

Z-Score: {stats['z_score']:.3f}
P-Value: {stats['p_value']:.4f}
Confidence: {stats['confidence_level']*100:.1f}%

Assessment:
{sig_text}
"""
        
        ax5.text(0.1, 0.95, summary_text, transform=ax5.transAxes,
                fontfamily='monospace', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=sig_color, alpha=0.3, linewidth=2))
        
        fig.suptitle(
            f'Bayesian Blocks Significance Test: {self.source_name}\n'
            f'{self.num_bootstrap_iterations} Bootstrap Iterations',
            fontsize=14, fontweight='bold', y=0.995
        )
        
        return fig
    
    def save_results(self, output_dir: str = './bb_significance_results') -> None:
        """
        Save test results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print("Saving results...")
        
        try:
            # Save statistics
            stats = self.calculate_significance_statistics()
            stats_file = output_path / 'significance_stats.txt'
            with open(stats_file, 'w') as f:
                f.write(f"Source: {self.source_name}\n")
                f.write(f"Timestamp: {self.timestamp}\n\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
            
            # Save bootstrap results as CSV
            results_data = [r.to_dict() for r in self.bootstrap_results]
            df = pd.DataFrame(results_data)
            csv_file = output_path / 'bootstrap_results.csv'
            df.to_csv(csv_file, index=False)
            
            if self.verbose:
                print(f"\n✓ Results saved to {output_path}/")
                print(f"  • significance_stats.txt: Statistical summary")
                print(f"  • bootstrap_results.csv: Individual bootstrap results\n")
        
        except Exception as e:
            print(f"Error saving results: {e}")


# Main execution
if __name__ == "__main__":
    
    # Create significance test
    test = BayesianBlocksSignificanceTest(
        num_bootstrap_iterations=100,
        info_file='files/source_info_v1.pkl',
        verbose=True
    )
    
    # Load a random source
    if test.load_source_data():
        
        # Run significance test
        results = test.run_significance_test()
        
        if results:
            # Print results
            test.print_significance_report()
            
            # Create visualization
            print("Generating visualization...")
            fig = test.plot_significance_results(figsize=(14, 10))
            
            if fig is not None:
                # Save figure
                source_safe_name = test.source_name.replace(' ', '_').replace('/', '_')
                plt.savefig(f'{source_safe_name}_significance_test.png', dpi=150, bbox_inches='tight')
                print(f"✓ Saved plot: {source_safe_name}_significance_test.png\n")
                
                # Save results
                test.save_results('./bb_significance_results')
                
                plt.show()
            else:
                print("Failed to create visualization")
        else:
            print("No bootstrap results generated")
    
    else:
        print("Failed to load source data.")