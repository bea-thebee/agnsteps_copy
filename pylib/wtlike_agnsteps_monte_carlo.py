"""
Monte Carlo Analysis of Bayesian Blocks Detection
Runs multiple simulations to characterize the distribution of detected blocks
across different AGN flux profiles and noise conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Import the simulator
from wtlike_agnsteps_simulator import WTLikeAGNSimulator


@dataclass
class MonteCarloResult:
    """Results from a single simulation run"""
    run_id: int
    num_photons: int
    num_cells: int
    num_blocks: int
    num_steps: int
    compression_ratio: float
    mean_cell_flux: float
    std_cell_flux: float
    block_edges: np.ndarray
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'run_id': self.run_id,
            'num_photons': self.num_photons,
            'num_cells': self.num_cells,
            'num_blocks': self.num_blocks,
            'num_steps': self.num_steps,
            'compression_ratio': float(self.compression_ratio),
            'mean_cell_flux': float(self.mean_cell_flux),
            'std_cell_flux': float(self.std_cell_flux),
        }


class MonteCarloSimulation:
    """
    Runs multiple WTLike AGN simulations and analyzes the distribution
    of detected Bayesian blocks.
    """
    
    def __init__(self, num_runs: int = 100, verbose: bool = True):
        """
        Initialize Monte Carlo simulation.
        
        Args:
            num_runs: Number of simulation runs to perform
            verbose: Print progress information
        """
        self.num_runs = num_runs
        self.verbose = verbose
        self.results: List[MonteCarloResult] = []
        self.simulation_config = None
        self.timestamp = datetime.now()
        
    def run_simulations(self, 
                       num_photons: int = 40000,
                       bin_width: float = 7.0,
                       p0: float = 0.05,
                       flux_steps: Optional[List[Dict]] = None,
                       mission_duration: float = 5475) -> List[MonteCarloResult]:
        """
        Run Monte Carlo simulations.
        
        Args:
            num_photons: Target number of photons per run
            bin_width: Time bin width in days
            p0: False positive probability for Bayesian Blocks
            flux_steps: List of flux step definitions. If None, uses default steps.
            mission_duration: Total mission duration in days
            
        Returns:
            List of MonteCarloResult objects
        """
        
        # Default flux steps if not provided
        if flux_steps is None:
            flux_steps = [
                {'start': 54683, 'end': 55500, 'rate': 1.0e-7, 'name': 'Low'},
                {'start': 55500, 'end': 55700, 'rate': 3.5e-7, 'name': 'High'},
                {'start': 55700, 'end': 57500, 'rate': 1.5e-7, 'name': 'Intermediate'},
            ]
        
        # Store configuration
        self.simulation_config = {
            'num_runs': self.num_runs,
            'num_photons': num_photons,
            'bin_width': bin_width,
            'p0': p0,
            'mission_duration': mission_duration,
            'num_flux_steps': len(flux_steps),
            'flux_steps': flux_steps,
        }
        
        self.results = []
        
        if self.verbose:
            print(f"Starting Monte Carlo Simulation: {self.num_runs} runs")
            print(f"  Photons per run: {num_photons:,}")
            print(f"  Bin width: {bin_width} days")
            print(f"  Bayesian Blocks p0: {p0}")
            print(f"  Flux steps: {len(flux_steps)}")
            print(f"  Mission duration: {mission_duration} days\n")
        
        for run_id in range(self.num_runs):
            if self.verbose and (run_id + 1) % max(1, self.num_runs // 10) == 0:
                print(f"  Progress: {run_id + 1}/{self.num_runs} runs completed")
            
            # Create simulator
            sim = WTLikeAGNSimulator(
                mission_duration=mission_duration,
                source_rate=2.0e-7,
                background_rate=3.5e-7,
                effective_area=2800
            )
            
            # Add flux steps
            for step in flux_steps:
                sim.add_flux_step(
                    step['start'],
                    step['end'],
                    step['rate'],
                    step.get('name', f"Step {len(sim.steps) + 1}")
                )
            
            # Generate photons with some randomness in the total count
            photons_this_run = int(num_photons * np.random.uniform(0.95, 1.05))
            photons = sim.generate_photons(num_photons_target=photons_this_run)
            
            # Create light curve
            start_time = flux_steps[0]['start']
            end_time = flux_steps[-1]['end']
            lc = sim.create_light_curve((start_time, end_time, bin_width))
            
            # Apply Bayesian Blocks (suppress verbose output)
            fitness = type(sim.apply_bayesian_blocks.__self__).__bases__[0]
            from wtlike_agnsteps_simulator import CountFitness
            fitness_obj = CountFitness(lc.cells, p0=p0)
            block_edges = fitness_obj.fit()
            num_blocks = len(block_edges) - 1
            
            # Collect statistics
            cells = lc.cells
            result = MonteCarloResult(
                run_id=run_id,
                num_photons=len(photons),
                num_cells=len(cells),
                num_blocks=num_blocks,
                num_steps=len(flux_steps),
                compression_ratio=len(cells) / num_blocks if num_blocks > 0 else 0,
                mean_cell_flux=cells['S'].mean(),
                std_cell_flux=cells['S'].std(),
                block_edges=block_edges
            )
            
            self.results.append(result)
        
        if self.verbose:
            print(f"\nMonte Carlo simulation complete!\n")
        
        return self.results
    
    def get_statistics(self) -> Dict:
        """
        Calculate statistics from Monte Carlo results.
        
        Returns:
            Dictionary with statistical summaries
        """
        if not self.results:
            raise ValueError("No results available. Run simulations first.")
        
        blocks = np.array([r.num_blocks for r in self.results])
        cells = np.array([r.num_cells for r in self.results])
        compression = np.array([r.compression_ratio for r in self.results])
        flux_mean = np.array([r.mean_cell_flux for r in self.results])
        flux_std = np.array([r.std_cell_flux for r in self.results])
        
        stats = {
            'num_blocks': {
                'mean': float(np.mean(blocks)),
                'median': float(np.median(blocks)),
                'std': float(np.std(blocks)),
                'min': int(np.min(blocks)),
                'max': int(np.max(blocks)),
                '95_ci': (float(np.percentile(blocks, 2.5)), 
                         float(np.percentile(blocks, 97.5))),
            },
            'num_cells': {
                'mean': float(np.mean(cells)),
                'median': float(np.median(cells)),
                'std': float(np.std(cells)),
            },
            'compression_ratio': {
                'mean': float(np.mean(compression)),
                'std': float(np.std(compression)),
            },
            'flux_statistics': {
                'mean_flux_mean': float(np.mean(flux_mean)),
                'mean_flux_std': float(np.mean(flux_std)),
            },
            'total_runs': self.num_runs,
        }
        
        return stats
    
    def print_summary(self) -> None:
        """Print formatted summary of Monte Carlo results"""
        stats = self.get_statistics()
        
        print("=" * 70)
        print("MONTE CARLO SIMULATION SUMMARY")
        print("=" * 70)
        print(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Configuration:")
        print(f"  Total runs: {self.num_runs}")
        print(f"  Photons per run: {self.simulation_config['num_photons']:,}")
        print(f"  Bin width: {self.simulation_config['bin_width']} days")
        print(f"  Bayesian Blocks p0: {self.simulation_config['p0']}")
        print(f"  Flux steps defined: {self.simulation_config['num_flux_steps']}\n")
        
        print(f"Bayesian Blocks Detected:")
        print(f"  Mean: {stats['num_blocks']['mean']:.2f}")
        print(f"  Median: {stats['num_blocks']['median']:.1f}")
        print(f"  Std Dev: {stats['num_blocks']['std']:.2f}")
        print(f"  Range: [{stats['num_blocks']['min']}, {stats['num_blocks']['max']}]")
        print(f"  95% CI: ({stats['num_blocks']['95_ci'][0]:.1f}, "
              f"{stats['num_blocks']['95_ci'][1]:.1f})\n")
        
        print(f"Time Cells:")
        print(f"  Mean: {stats['num_cells']['mean']:.1f}")
        print(f"  Std Dev: {stats['num_cells']['std']:.1f}\n")
        
        print(f"Compression Ratio (cells/blocks):")
        print(f"  Mean: {stats['compression_ratio']['mean']:.2f}x")
        print(f"  Std Dev: {stats['compression_ratio']['std']:.2f}\n")
        
        print("=" * 70)
    
    def plot_results(self, figsize: Tuple[float, float] = (16, 12)) -> plt.Figure:
        """
        Create comprehensive visualization of Monte Carlo results.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        if not self.results:
            raise ValueError("No results available. Run simulations first.")
        
        blocks = np.array([r.num_blocks for r in self.results])
        cells = np.array([r.num_cells for r in self.results])
        compression = np.array([r.compression_ratio for r in self.results])
        flux_mean = np.array([r.mean_cell_flux for r in self.results])
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. Histogram of blocks detected
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(blocks, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(blocks), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(blocks):.2f}')
        ax1.set_xlabel('Number of Blocks Detected')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Bayesian Blocks')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Time series of blocks
        ax2 = fig.add_subplot(gs[0, 1])
        run_ids = np.arange(self.num_runs)
        ax2.plot(run_ids, blocks, 'o-', markersize=4, alpha=0.6)
        ax2.axhline(np.mean(blocks), color='red', linestyle='--', linewidth=2, 
                   label='Mean')
        ax2.fill_between(run_ids, 
                        np.mean(blocks) - np.std(blocks),
                        np.mean(blocks) + np.std(blocks),
                        alpha=0.2, color='red', label='±1 σ')
        ax2.set_xlabel('Run ID')
        ax2.set_ylabel('Number of Blocks')
        ax2.set_title('Blocks per Run (Time Series)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot (normality check)
        ax3 = fig.add_subplot(gs[0, 2])
        from scipy import stats as sp_stats
        sp_stats.probplot(blocks, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normality Check)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Blocks vs Cells
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(cells, blocks, alpha=0.6, s=50)
        z = np.polyfit(cells, blocks, 1)
        p = np.poly1d(z)
        ax4.plot(cells, p(cells), "r--", linewidth=2)
        ax4.set_xlabel('Number of Cells')
        ax4.set_ylabel('Number of Blocks')
        ax4.set_title(f'Blocks vs Cells (r={np.corrcoef(cells, blocks)[0,1]:.3f})')
        ax4.grid(True, alpha=0.3)
        
        # 5. Compression ratio histogram
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(compression, bins=15, color='coral', edgecolor='black', alpha=0.7)
        ax5.axvline(np.mean(compression), color='darkred', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(compression):.2f}x')
        ax5.set_xlabel('Compression Ratio (cells/blocks)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of Compression Ratios')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Box plot of blocks
        ax6 = fig.add_subplot(gs[1, 2])
        bp = ax6.boxplot([blocks], labels=['Bayesian Blocks'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax6.set_ylabel('Number of Blocks')
        ax6.set_title('Box Plot: Blocks Detected')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Cumulative distribution
        ax7 = fig.add_subplot(gs[2, 0])
        sorted_blocks = np.sort(blocks)
        cumulative = np.arange(1, len(sorted_blocks) + 1) / len(sorted_blocks)
        ax7.plot(sorted_blocks, cumulative, 'o-', markersize=5, alpha=0.7)
        ax7.set_xlabel('Number of Blocks')
        ax7.set_ylabel('Cumulative Probability')
        ax7.set_title('Cumulative Distribution Function')
        ax7.grid(True, alpha=0.3)
        
        # 8. Mean flux vs blocks
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.scatter(flux_mean, blocks, alpha=0.6, s=50, c=compression, cmap='viridis')
        cbar = plt.colorbar(ax8.collections[0], ax=ax8)
        cbar.set_label('Compression Ratio')
        ax8.set_xlabel('Mean Cell Flux')
        ax8.set_ylabel('Number of Blocks')
        ax8.set_title('Blocks vs Mean Flux')
        ax8.grid(True, alpha=0.3)
        
        # 9. Statistical summary text box
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        stats = self.get_statistics()
        summary_text = (
            f"STATISTICS\n"
            f"{'─' * 30}\n"
            f"Blocks Detected:\n"
            f"  Mean: {stats['num_blocks']['mean']:.2f} ± {stats['num_blocks']['std']:.2f}\n"
            f"  Median: {stats['num_blocks']['median']:.0f}\n"
            f"  Range: [{stats['num_blocks']['min']}, {stats['num_blocks']['max']}]\n"
            f"  95% CI: ({stats['num_blocks']['95_ci'][0]:.1f}, "
            f"{stats['num_blocks']['95_ci'][1]:.1f})\n\n"
            f"Compression:\n"
            f"  Mean: {stats['compression_ratio']['mean']:.2f}x\n"
            f"  Std Dev: {stats['compression_ratio']['std']:.2f}\n\n"
            f"Cells per Run:\n"
            f"  Mean: {stats['num_cells']['mean']:.0f}\n"
            f"  Std Dev: {stats['num_cells']['std']:.1f}\n"
        )
        ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
                fontfamily='monospace', fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(
            f'Monte Carlo Analysis: {self.num_runs} AGN Simulation Runs\n'
            f'Bayesian Blocks Detection Statistics',
            fontsize=14, fontweight='bold', y=0.995
        )
        
        return fig
    
    def save_results(self, output_dir: str = './monte_carlo_results') -> None:
        """
        Save Monte Carlo results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_file = output_path / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(self.simulation_config, f, indent=2)
        
        # Save individual results
        results_data = [r.to_dict() for r in self.results]
        results_file = output_path / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save statistics
        stats = self.get_statistics()
        stats_file = output_path / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame([r.to_dict() for r in self.results])
        csv_file = output_path / 'results.csv'
        df.to_csv(csv_file, index=False)
        
        if self.verbose:
            print(f"Results saved to {output_path}/")
            print(f"  - config.json: Simulation configuration")
            print(f"  - results.json: Individual run results")
            print(f"  - statistics.json: Summary statistics")
            print(f"  - results.csv: Results in CSV format")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        return pd.DataFrame([r.to_dict() for r in self.results])


# Example usage
if __name__ == "__main__":
    
    # Create Monte Carlo simulation
    mc = MonteCarloSimulation(num_runs=100, verbose=True)
    
    # Define AGN flux profile (3 distinct steps)
    flux_profile = [
        {'start': 54683, 'end': 55500, 'rate': 1.0e-7, 'name': 'Low State'},
        {'start': 55500, 'end': 55700, 'rate': 3.5e-7, 'name': 'High State'},
        {'start': 55700, 'end': 57500, 'rate': 1.5e-7, 'name': 'Intermediate'},
    ]
    
    # Run simulations
    results = mc.run_simulations(
        num_photons=40000,
        bin_width=7.0,
        p0=0.05,
        flux_steps=flux_profile,
        mission_duration=5475
    )
    
    # Print summary statistics
    mc.print_summary()
    
    # Create visualization
    print("Creating visualization...")
    fig = mc.plot_results(figsize=(16, 12))
    plt.tight_layout()
    plt.savefig('monte_carlo_results.png', dpi=150, bbox_inches='tight')
    print("Saved to monte_carlo_results.png\n")
    
    # Save results to files
    mc.save_results('./monte_carlo_results')
    
    # Optional: Access results as DataFrame for further analysis
    df = mc.to_dataframe()
    print("\nResults DataFrame:")
    print(df.describe())
    
    plt.show()