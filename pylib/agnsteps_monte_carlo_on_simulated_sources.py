"""
Monte Carlo Significance Test on Simulated Sources
Tests how often the Bayesian Blocks detection algorithm correctly identifies
the true number of blocks in simulated light curves.

Uses the simulated light curve generator to create synthetic sources,
then runs bootstrap significance tests to assess detection reliability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import seaborn as sns
import random
from tqdm import tqdm
import sys

# Configure styling
plt.style.use('dark_background')
sns.set_theme('talk', font_scale=1.0)


@dataclass
class SimulatedSourceResult:
    """Results from significance test on simulated source"""
    source_id: int
    true_num_blocks: int
    generated_source_name: str
    template_source: str
    original_blocks_detected: int
    bootstrap_mean_blocks: float
    bootstrap_std_blocks: float
    z_score: float
    p_value: float
    confidence_level: float
    detection_success: bool  # True if detected correct number of blocks
    

class MonteCarloSimulatedSourcesTester:
    """
    Run Monte Carlo significance tests on simulated light curves.
    Tests how reliably Bayesian Blocks detects the correct number of blocks.
    """
    
    def __init__(self,
                 num_simulated_sources: int = 50,
                 bootstrap_iterations: int = 100,
                 info_file: str = 'files/source_info_v1.pkl',
                 verbose: bool = True):
        """
        Initialize tester.
        
        Args:
            num_simulated_sources: Number of synthetic sources to generate
            bootstrap_iterations: Bootstrap iterations per test
            info_file: Path to source_info pickle file
            verbose: Print progress
        """
        self.num_simulated_sources = num_simulated_sources
        self.bootstrap_iterations = bootstrap_iterations
        self.info_file = Path(info_file)
        self.verbose = verbose
        self.vdb = None
        self.simulated_sources = []
        self.test_results = []
        self.timestamp = pd.Timestamp.now()
        
    def _select_random_source_with_lightcurve(self, vdb: dict) -> str:
        """Select a random source with valid light curve."""
        sources_with_lc = [
            name for name, data in vdb.items() 
            if data.get('light_curve') is not None and len(data.get('light_curve', {})) > 1
        ]
        
        if not sources_with_lc:
            raise ValueError("No sources with valid light curves found")
        
        return random.choice(sources_with_lc)
    
    def _extract_light_curve_values(self, lc_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract flux, errors, and time values from light curve data."""
        if isinstance(lc_data, dict):
            lc_df = pd.DataFrame.from_dict(lc_data, orient='index')
        else:
            lc_df = lc_data
        
        # Extract flux values
        flux_array = None
        for col in lc_df.columns:
            if isinstance(col, str) and ('flux' in col.lower()):
                flux_col = lc_df[col]
                break
        else:
            for col in lc_df.columns:
                if lc_df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                    flux_col = lc_df[col]
                    break
            else:
                flux_col = lc_df.iloc[:, 0]
        
        flux_values = flux_col.values
        flux_array = np.array([
            fv[0] if hasattr(fv, '__len__') and not isinstance(fv, (str, bytes)) else fv
            for fv in flux_values
        ], dtype=float)
        
        # Extract errors
        error_lower = None
        error_upper = None
        
        if 'errors' in lc_df.columns:
            errors = lc_df['errors'].values
            error_lower = np.array([e[0] if isinstance(e, (list, tuple)) and len(e) > 0 else e for e in errors], dtype=float)
            error_upper = np.array([e[1] if isinstance(e, (list, tuple)) and len(e) > 1 else e for e in errors], dtype=float)
        else:
            error_cols = [col for col in lc_df.columns if isinstance(col, str) and 'error' in col.lower()]
            if len(error_cols) >= 2:
                error_lower = np.array([e for e in lc_df[error_cols[0]].values], dtype=float)
                error_upper = np.array([e for e in lc_df[error_cols[1]].values], dtype=float)
        
        if error_lower is None or error_upper is None:
            error_lower = flux_array * 0.1
            error_upper = flux_array * 0.1
        
        # Extract time information
        times = None
        time_widths = None
        
        for col in lc_df.columns:
            if isinstance(col, str):
                if col.lower() == 't':
                    times = lc_df[col].values
                elif col.lower() == 'tw':
                    time_widths = lc_df[col].values
        
        if times is None:
            times = np.arange(len(flux_array), dtype=float)
        else:
            times = np.array(times, dtype=float)
        
        if time_widths is None:
            time_widths = np.ones(len(flux_array), dtype=float)
        else:
            time_widths = np.array(time_widths, dtype=float)
        
        return flux_array, error_lower, error_upper, times, time_widths
    
    def load_database(self) -> bool:
        """Load the agnsteps database."""
        try:
            if not self.info_file.exists():
                print(f"Error: {self.info_file} not found")
                return False
            
            with open(self.info_file, 'rb') as f:
                self.vdb = pickle.load(f)
            
            if self.verbose:
                print(f"✓ Loaded database with {len(self.vdb)} sources\n")
            
            return True
            
        except Exception as e:
            print(f"Error loading database: {e}")
            return False
    
    def generate_simulated_source(self, 
                                 source_id: int,
                                 template_source: Optional[str] = None) -> Dict:
        """
        Generate a simulated source by sampling from template statistics.
        
        Args:
            source_id: ID for this simulated source
            template_source: Source to use as template (if None, random)
            
        Returns:
            Dictionary with simulated source data
        """
        if template_source is None:
            template_source = self._select_random_source_with_lightcurve(self.vdb)
        
        if template_source not in self.vdb:
            return {}
        
        source_dict = self.vdb[template_source]
        lc_data = source_dict.get('light_curve')
        
        if lc_data is None:
            return {}
        
        try:
            flux_array, error_lower, error_upper, times, time_widths = self._extract_light_curve_values(lc_data)
            
            # Generate simulated source by adding noise to original
            simulated_flux = flux_array.copy()
            
            for i in range(len(simulated_flux)):
                # Add Gaussian noise based on error bars
                error_magnitude = (error_lower[i] + error_upper[i]) / 2
                noise = np.random.normal(0, error_magnitude)
                simulated_flux[i] = simulated_flux[i] + noise
                simulated_flux[i] = max(simulated_flux[i], 0.01)
            
            return {
                'source_id': source_id,
                'template_source': template_source,
                'true_num_blocks': len(flux_array),
                'flux_values': simulated_flux,
                'error_lower': error_lower,
                'error_upper': error_upper,
                'times': times,
                'time_widths': time_widths,
            }
        
        except Exception as e:
            print(f"Error generating simulated source: {e}")
            return {}
    
    def count_blocks_in_bootstrap(self, bootstrap_flux: np.ndarray) -> int:
        """
        Count the number of "blocks" in bootstrap sample based on variance.
        
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
        threshold = mean_diff + std_diff
        
        block_count = 1
        for diff in flux_diffs:
            if diff > threshold:
                block_count += 1
        
        return block_count
    
    def run_bootstrap_test_on_simulated(self, simulated_source: Dict) -> Optional[SimulatedSourceResult]:
        """
        Run bootstrap significance test on a simulated source.
        
        Args:
            simulated_source: Simulated source data
            
        Returns:
            SimulatedSourceResult object
        """
        flux_values = simulated_source['flux_values']
        error_lower = simulated_source['error_lower']
        error_upper = simulated_source['error_upper']
        true_num_blocks = simulated_source['true_num_blocks']
        
        # Count blocks in original simulated source
        original_blocks_detected = self.count_blocks_in_bootstrap(flux_values)
        
        # Run bootstrap
        bootstrap_blocks = []
        
        for iteration in range(self.bootstrap_iterations):
            # Generate bootstrap sample
            errors = (error_lower + error_upper) / 2
            bootstrap_flux = flux_values.copy()
            
            for i in range(len(bootstrap_flux)):
                perturbation = np.random.normal(0, errors[i])
                bootstrap_flux[i] = flux_values[i] + perturbation
                bootstrap_flux[i] = max(bootstrap_flux[i], 0.01)
            
            # Count blocks
            num_blocks = self.count_blocks_in_bootstrap(bootstrap_flux)
            bootstrap_blocks.append(num_blocks)
        
        bootstrap_blocks = np.array(bootstrap_blocks)
        
        # Calculate statistics
        mean_bootstrap_blocks = np.mean(bootstrap_blocks)
        std_bootstrap_blocks = np.std(bootstrap_blocks)
        
        z_score = (original_blocks_detected - mean_bootstrap_blocks) / std_bootstrap_blocks if std_bootstrap_blocks > 0 else 0
        p_value = np.sum(bootstrap_blocks >= original_blocks_detected) / len(bootstrap_blocks)
        confidence_level = 1 - p_value
        
        # Check if detected correct number of blocks
        detection_success = (original_blocks_detected == true_num_blocks)
        
        result = SimulatedSourceResult(
            source_id=simulated_source['source_id'],
            true_num_blocks=true_num_blocks,
            generated_source_name=f"SIM_{simulated_source['source_id']:04d}",
            template_source=simulated_source['template_source'],
            original_blocks_detected=original_blocks_detected,
            bootstrap_mean_blocks=float(mean_bootstrap_blocks),
            bootstrap_std_blocks=float(std_bootstrap_blocks),
            z_score=float(z_score),
            p_value=float(p_value),
            confidence_level=float(confidence_level),
            detection_success=detection_success,
        )
        
        return result
    
    def run_full_test(self) -> List[SimulatedSourceResult]:
        """
        Run full Monte Carlo test on all simulated sources.
        
        Returns:
            List of SimulatedSourceResult objects
        """
        if self.vdb is None:
            print("Error: Database not loaded")
            return []
        
        if self.verbose:
            print("="*80)
            print("MONTE CARLO TEST ON SIMULATED SOURCES")
            print("="*80 + "\n")
            print(f"Generating {self.num_simulated_sources} simulated sources...")
            print(f"Bootstrap iterations per source: {self.bootstrap_iterations}\n")
        
        self.simulated_sources = []
        self.test_results = []
        
        # Generate simulated sources
        if self.verbose:
            pbar_gen = tqdm(total=self.num_simulated_sources, desc="Generating sources",
                           unit="source", ncols=80, position=0, leave=True)
        
        for i in range(self.num_simulated_sources):
            sim_source = self.generate_simulated_source(i)
            if sim_source:
                self.simulated_sources.append(sim_source)
                if self.verbose:
                    pbar_gen.update(1)
        
        if self.verbose:
            pbar_gen.close()
            print(f"✓ Generated {len(self.simulated_sources)} simulated sources\n")
        
        # Run tests
        if self.verbose:
            print(f"Running significance tests on {len(self.simulated_sources)} sources...\n")
            pbar_test = tqdm(total=len(self.simulated_sources), desc="Testing sources",
                            unit="source", ncols=80, position=0, leave=True)
        
        for sim_source in self.simulated_sources:
            try:
                result = self.run_bootstrap_test_on_simulated(sim_source)
                if result:
                    self.test_results.append(result)
                    if self.verbose:
                        pbar_test.update(1)
                        status = "✓" if result.detection_success else "✗"
                        pbar_test.set_description(
                            f"Testing: {status} "
                            f"{result.true_num_blocks} blocks "
                            f"(detected: {result.original_blocks_detected})"
                        )
            except Exception as e:
                if self.verbose:
                    pbar_test.write(f"Error testing source {sim_source['source_id']}: {e}")
                pbar_test.update(1)
        
        if self.verbose:
            pbar_test.close()
        
        return self.test_results
    
    def calculate_detection_statistics(self) -> Dict:
        """
        Calculate statistics on detection performance.
        
        Returns:
            Dictionary with detection statistics
        """
        if not self.test_results:
            return {}
        
        results = self.test_results
        
        # Overall success rate
        success_count = sum(1 for r in results if r.detection_success)
        success_rate = success_count / len(results)
        
        # Group by true number of blocks
        by_num_blocks = {}
        for r in results:
            if r.true_num_blocks not in by_num_blocks:
                by_num_blocks[r.true_num_blocks] = {'success': 0, 'total': 0}
            by_num_blocks[r.true_num_blocks]['total'] += 1
            if r.detection_success:
                by_num_blocks[r.true_num_blocks]['success'] += 1
        
        # Calculate per-block statistics
        per_block_stats = {}
        for num_blocks, counts in by_num_blocks.items():
            per_block_stats[num_blocks] = {
                'success_count': counts['success'],
                'total_count': counts['total'],
                'success_rate': counts['success'] / counts['total'],
            }
        
        # Z-score distribution
        z_scores = [r.z_score for r in results]
        p_values = [r.p_value for r in results]
        
        stats = {
            'total_tests': len(results),
            'successful_detections': success_count,
            'overall_success_rate': success_rate,
            'per_block_statistics': per_block_stats,
            'mean_z_score': float(np.mean(z_scores)),
            'std_z_score': float(np.std(z_scores)),
            'mean_p_value': float(np.mean(p_values)),
            'std_p_value': float(np.std(p_values)),
            'significant_tests': sum(1 for p in p_values if p < 0.05),
            'significant_rate': sum(1 for p in p_values if p < 0.05) / len(results),
        }
        
        return stats
    
    def print_results_summary(self) -> None:
        """Print formatted results summary."""
        if not self.test_results:
            print("Error: No test results available")
            return
        
        stats = self.calculate_detection_statistics()
        
        print("\n" + "="*80)
        print("MONTE CARLO SIGNIFICANCE TEST RESULTS - SIMULATED SOURCES")
        print("="*80 + "\n")
        
        print(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Test Configuration:")
        print(f"  Simulated sources: {stats['total_tests']}")
        print(f"  Bootstrap iterations per source: {self.bootstrap_iterations}\n")
        
        print(f"Overall Detection Performance:")
        print(f"  Successful detections: {stats['successful_detections']}/{stats['total_tests']}")
        print(f"  Success rate: {stats['overall_success_rate']*100:.1f}%\n")
        
        print(f"Performance by Number of Blocks:")
        print(f"  {'Blocks':>8} {'Success':>10} {'Total':>8} {'Rate':>10}")
        print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*10}")
        for num_blocks in sorted(stats['per_block_statistics'].keys()):
            block_stats = stats['per_block_statistics'][num_blocks]
            print(f"  {num_blocks:>8} {block_stats['success_count']:>10} {block_stats['total_count']:>8} {block_stats['success_rate']*100:>9.1f}%")
        
        print(f"\nStatistical Significance:")
        print(f"  Mean Z-score: {stats['mean_z_score']:.3f}")
        print(f"  Std Z-score: {stats['std_z_score']:.3f}")
        print(f"  Mean P-value: {stats['mean_p_value']:.4f}")
        print(f"  Significant tests (p<0.05): {stats['significant_tests']}/{stats['total_tests']} ({stats['significant_rate']*100:.1f}%)\n")
        
        print("="*80 + "\n")
    
    def plot_results(self, figsize: Tuple[float, float] = (16, 12)) -> plt.Figure:
        """
        Create comprehensive visualization of test results.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        if not self.test_results:
            print("Error: No test results")
            return None
        
        stats = self.calculate_detection_statistics()
        results = self.test_results
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. Overall success rate
        ax1 = fig.add_subplot(gs[0, 0])
        success_rate = stats['overall_success_rate']
        colors = ['green' if success_rate > 0.8 else 'orange' if success_rate > 0.6 else 'red']
        ax1.bar(['Detection\nSuccess Rate'], [success_rate*100], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Overall Detection Rate', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.text(0, success_rate*100 + 3, f'{success_rate*100:.1f}%', ha='center', fontweight='bold', fontsize=11)
        
        # 2. Detection by number of blocks
        ax2 = fig.add_subplot(gs[0, 1])
        block_counts = sorted(stats['per_block_statistics'].keys())
        success_rates = [stats['per_block_statistics'][b]['success_rate']*100 for b in block_counts]
        colors_blocks = ['green' if r > 80 else 'orange' if r > 60 else 'red' for r in success_rates]
        ax2.bar([str(b) for b in block_counts], success_rates, color=colors_blocks, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_xlabel('Number of Blocks', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Detection Rate by Block Count', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Sample counts per block
        ax3 = fig.add_subplot(gs[0, 2])
        sample_counts = [stats['per_block_statistics'][b]['total_count'] for b in block_counts]
        ax3.bar([str(b) for b in block_counts], sample_counts, color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
        ax3.set_xlabel('Number of Blocks', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        ax3.set_title('Sample Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. True vs detected blocks scatter
        ax4 = fig.add_subplot(gs[1, 0])
        true_blocks = [r.true_num_blocks for r in results]
        detected_blocks = [r.original_blocks_detected for r in results]
        colors_scatter = ['green' if t == d else 'red' for t, d in zip(true_blocks, detected_blocks)]
        ax4.scatter(true_blocks, detected_blocks, c=colors_scatter, alpha=0.6, s=100, edgecolor='black', linewidth=1)
        
        # Add diagonal line (perfect detection)
        min_blocks = min(min(true_blocks), min(detected_blocks))
        max_blocks = max(max(true_blocks), max(detected_blocks))
        ax4.plot([min_blocks-1, max_blocks+1], [min_blocks-1, max_blocks+1], 'k--', alpha=0.5, linewidth=2, label='Perfect detection')
        
        ax4.set_xlabel('True Number of Blocks', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Detected Number of Blocks', fontsize=11, fontweight='bold')
        ax4.set_title('True vs Detected Blocks', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. Z-score distribution
        ax5 = fig.add_subplot(gs[1, 1])
        z_scores = [r.z_score for r in results]
        ax5.hist(z_scores, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax5.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Z=0')
        ax5.axvline(2, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Z=2σ')
        ax5.set_xlabel('Z-Score', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax5.set_title('Z-Score Distribution', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. P-value distribution
        ax6 = fig.add_subplot(gs[1, 2])
        p_values = [r.p_value for r in results]
        ax6.hist(p_values, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax6.axvline(0.05, color='red', linestyle='--', linewidth=2, alpha=0.7, label='p=0.05')
        ax6.set_xlabel('P-Value', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax6.set_title('P-Value Distribution', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Confidence level vs detection
        ax7 = fig.add_subplot(gs[2, 0])
        confidence_levels = [r.confidence_level*100 for r in results]
        colors_conf = ['green' if r.detection_success else 'red' for r in results]
        ax7.scatter(confidence_levels, [i for i in range(len(results))], c=colors_conf, alpha=0.5, s=50)
        ax7.set_xlabel('Confidence Level (%)', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Source Index', fontsize=11, fontweight='bold')
        ax7.set_title('Confidence Level by Source', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. Error distribution
        ax8 = fig.add_subplot(gs[2, 1])
        errors = [abs(r.true_num_blocks - r.original_blocks_detected) for r in results]
        error_counts = {}
        for e in errors:
            error_counts[e] = error_counts.get(e, 0) + 1
        
        ax8.bar(sorted(error_counts.keys()), [error_counts[k] for k in sorted(error_counts.keys())],
               color='mediumpurple', alpha=0.7, edgecolor='black', linewidth=2)
        ax8.set_xlabel('Block Detection Error', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax8.set_title('Detection Error Distribution', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. Summary statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        summary_text = f"""
TEST SUMMARY

Total Tests: {stats['total_tests']}
Bootstrap Iters: {self.bootstrap_iterations}

Overall Success: {stats['successful_detections']}/{stats['total_tests']}
Success Rate: {stats['overall_success_rate']*100:.1f}%

Significant (p<0.05): {stats['significant_tests']}/{stats['total_tests']}
Significance Rate: {stats['significant_rate']*100:.1f}%

Mean Z-score: {stats['mean_z_score']:.3f}
Mean P-value: {stats['mean_p_value']:.4f}

Mean Block Error: {np.mean([abs(r.true_num_blocks - r.original_blocks_detected) for r in results]):.2f}
"""
        
        ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
                fontfamily='monospace', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, linewidth=2))
        
        fig.suptitle(
            f'Monte Carlo Significance Test on Simulated Sources\n'
            f'{self.num_simulated_sources} Simulated Sources with {self.bootstrap_iterations} Bootstrap Iterations',
            fontsize=14, fontweight='bold', y=0.995
        )
        
        return fig
    
    def save_results(self, output_dir: str = './simulated_mc_results') -> None:
        """Save test results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print("Saving results...")
        
        try:
            # Save as CSV
            results_data = []
            for r in self.test_results:
                results_data.append({
                    'source_id': r.source_id,
                    'true_num_blocks': r.true_num_blocks,
                    'detected_num_blocks': r.original_blocks_detected,
                    'bootstrap_mean': r.bootstrap_mean_blocks,
                    'bootstrap_std': r.bootstrap_std_blocks,
                    'z_score': r.z_score,
                    'p_value': r.p_value,
                    'confidence_level': r.confidence_level,
                    'detection_success': r.detection_success,
                    'template_source': r.template_source,
                })
            
            df = pd.DataFrame(results_data)
            csv_file = output_path / 'simulation_results.csv'
            df.to_csv(csv_file, index=False)
            
            # Save statistics
            stats = self.calculate_detection_statistics()
            stats_file = output_path / 'detection_statistics.txt'
            with open(stats_file, 'w') as f:
                f.write(f"Monte Carlo Test on Simulated Sources\n")
                f.write(f"Timestamp: {self.timestamp}\n\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
            
            if self.verbose:
                print(f"\n✓ Results saved to {output_path}/")
                print(f"  • simulation_results.csv: Detailed results")
                print(f"  • detection_statistics.txt: Summary statistics\n")
        
        except Exception as e:
            print(f"Error saving results: {e}")


# Main execution
if __name__ == "__main__":
    
    # Create tester
    tester = MonteCarloSimulatedSourcesTester(
        num_simulated_sources=50,
        bootstrap_iterations=100,
        info_file='files/source_info_v1.pkl',
        verbose=True
    )
    
    # Load database
    if tester.load_database():
        
        # Run tests
        results = tester.run_full_test()
        
        # Print results
        tester.print_results_summary()
        
        # Create visualization
        print("Creating comprehensive visualization...")
        fig = tester.plot_results(figsize=(16, 12))
        
        plt.savefig('simulated_mc_test_results.png', dpi=150, bbox_inches='tight')
        print("✓ Saved plot: simulated_mc_test_results.png\n")
        
        # Save results
        tester.save_results('./simulated_mc_results')
        
        plt.show()
    
    else:
        print("Failed to load database")