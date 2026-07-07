"""
Multi-Bin-Width Monte Carlo Analysis of Bayesian Blocks Detection
Runs parallel Monte Carlo simulations across different temporal bin widths
(7, 30, 180, 360 days) to study how temporal resolution affects Bayesian Block detection.

This allows systematic investigation of how different observational cadences
impact step/block detection in AGN light curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import sys

# Import the simulator
from wtlike_agnsteps_simulator import WTLikeAGNSimulator, CountFitness


@dataclass
class MonteCarloResult:
    """Results from a single simulation run"""
    run_id: int
    bin_width: float
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
            'bin_width': self.bin_width,
            'num_photons': self.num_photons,
            'num_cells': self.num_cells,
            'num_blocks': self.num_blocks,
            'num_steps': self.num_steps,
            'compression_ratio': float(self.compression_ratio),
            'mean_cell_flux': float(self.mean_cell_flux),
            'std_cell_flux': float(self.std_cell_flux),
        }


class MultiBindWidthMonteCarloSimulation:
    """
    Runs parallel Monte Carlo simulations across multiple bin widths.
    Compares how Bayesian Blocks detection varies with temporal resolution.
    """
    
    def __init__(self, 
                 bin_widths: List[float] = [7, 30, 180, 360],
                 num_runs_per_bin: int = 50,
                 verbose: bool = True):
        """
        Initialize multi-bin-width Monte Carlo simulation.
        
        Args:
            bin_widths: List of bin widths in days to simulate
            num_runs_per_bin: Number of simulation runs per bin width
            verbose: Print progress information
        """
        self.bin_widths = sorted(bin_widths)
        self.num_runs_per_bin = num_runs_per_bin
        self.total_runs = len(bin_widths) * num_runs_per_bin
        self.verbose = verbose
        self.results_by_bin: Dict[float, List[MonteCarloResult]] = {}
        self.simulation_config = None
        self.timestamp = datetime.now()
        
    def run_all_simulations(self, 
                           num_photons: int = 40000,
                           p0: float = 0.05,
                           flux_steps: Optional[List[Dict]] = None,
                           mission_duration: float = 5475) -> Dict[float, List[MonteCarloResult]]:
        """
        Run Monte Carlo simulations across all bin widths.
        
        Args:
            num_photons: Target number of photons per run
            p0: False positive probability for Bayesian Blocks
            flux_steps: List of flux step definitions
            mission_duration: Total mission duration in days
            
        Returns:
            Dictionary mapping bin_width → list of MonteCarloResult objects
        """
        
        # Default flux steps if not provided
        if flux_steps is None:
            flux_steps = [
                {'start': 54683, 'end': 55500, 'rate': 1.0e-7, 'name': 'Low State'},
                {'start': 55500, 'end': 55700, 'rate': 3.5e-7, 'name': 'High State'},
                {'start': 55700, 'end': 57500, 'rate': 1.5e-7, 'name': 'Intermediate'},
            ]
        
        # Store configuration
        self.simulation_config = {
            'num_runs_per_bin': self.num_runs_per_bin,
            'bin_widths': self.bin_widths,
            'total_runs': self.total_runs,
            'num_photons': num_photons,
            'p0': p0,
            'mission_duration': mission_duration,
            'num_flux_steps': len(flux_steps),
            'flux_steps': flux_steps,
        }
        
        self.results_by_bin = {bw: [] for bw in self.bin_widths}
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Multi-Bin-Width Monte Carlo Simulation")
            print(f"{'='*70}")
            print(f"Configuration:")
            print(f"  Bin widths: {self.bin_widths} days")
            print(f"  Runs per bin: {self.num_runs_per_bin}")
            print(f"  Total runs: {self.total_runs}")
            print(f"  Photons per run: {num_photons:,}")
            print(f"  Bayesian Blocks p0: {p0}")
            print(f"  Flux steps: {len(flux_steps)}")
            print(f"  Mission duration: {mission_duration} days")
            print(f"{'='*70}\n")
        
        # Main loop across bin widths
        overall_pbar = tqdm(
            total=self.total_runs,
            desc="Overall progress",
            unit="run",
            ncols=80,
            position=0,
            leave=True,
            file=sys.stdout
        )
        
        for bin_width in self.bin_widths:
            if self.verbose:
                print(f"\n{'─'*70}")
                print(f"Processing bin width: {bin_width} days")
                print(f"{'─'*70}")
            
            # Inner progress bar for this bin width
            bin_pbar = tqdm(
                total=self.num_runs_per_bin,
                desc=f"Bin {bin_width}d",
                unit="run",
                ncols=70,
                position=1,
                leave=False,
                file=sys.stdout
            )
            
            for run_id in range(self.num_runs_per_bin):
                try:
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
                    
                    # Update progress
                    bin_pbar.set_description(f"Bin {bin_width}d: Generating photons (Run {run_id + 1}/{self.num_runs_per_bin})")
                    
                    # Generate photons with randomness
                    photons_this_run = int(num_photons * np.random.uniform(0.95, 1.05))
                    photons = sim.generate_photons(num_photons_target=photons_this_run)
                    
                    bin_pbar.set_description(f"Bin {bin_width}d: Creating light curve (Run {run_id + 1}/{self.num_runs_per_bin})")
                    
                    # Create light curve with THIS bin width
                    start_time = flux_steps[0]['start']
                    end_time = flux_steps[-1]['end']
                    lc = sim.create_light_curve((start_time, end_time, bin_width))
                    
                    bin_pbar.set_description(f"Bin {bin_width}d: Applying Bayesian Blocks (Run {run_id + 1}/{self.num_runs_per_bin})")
                    
                    # Apply Bayesian Blocks
                    fitness_obj = CountFitness(lc.cells, p0=p0)
                    block_edges = fitness_obj.fit()
                    num_blocks = len(block_edges) - 1
                    
                    # Collect statistics
                    cells = lc.cells
                    result = MonteCarloResult(
                        run_id=run_id,
                        bin_width=bin_width,
                        num_photons=len(photons),
                        num_cells=len(cells),
                        num_blocks=num_blocks,
                        num_steps=len(flux_steps),
                        compression_ratio=len(cells) / num_blocks if num_blocks > 0 else 0,
                        mean_cell_flux=cells['S'].mean(),
                        std_cell_flux=cells['S'].std(),
                        block_edges=block_edges
                    )
                    
                    self.results_by_bin[bin_width].append(result)
                    
                    # Update progress bars
                    bin_pbar.update(1)
                    overall_pbar.update(1)
                    
                except Exception as e:
                    bin_pbar.write(f"Error in bin {bin_width}d, run {run_id + 1}: {str(e)}")
                    bin_pbar.update(1)
                    overall_pbar.update(1)
                    continue
            
            bin_pbar.close()
        
        overall_pbar.close()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Monte Carlo simulation complete!")
            print(f"Summary:")
            for bw in self.bin_widths:
                n_successful = len(self.results_by_bin[bw])
                print(f"  {bw:>3}d bins: {n_successful:>3}/{self.num_runs_per_bin} successful runs")
            print(f"{'='*70}\n")
        
        return self.results_by_bin
    
    def get_statistics_all_bins(self) -> Dict[float, Dict]:
        """
        Calculate statistics for all bin widths.
        
        Returns:
            Dictionary mapping bin_width → statistics dict
        """
        stats_by_bin = {}
        
        for bin_width in self.bin_widths:
            if bin_width not in self.results_by_bin:
                continue
            
            results = self.results_by_bin[bin_width]
            if not results:
                continue
            
            blocks = np.array([r.num_blocks for r in results])
            cells = np.array([r.num_cells for r in results])
            compression = np.array([r.compression_ratio for r in results])
            flux_mean = np.array([r.mean_cell_flux for r in results])
            
            stats_by_bin[bin_width] = {
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
                    'mean_flux_std': float(np.mean([r.std_cell_flux for r in results])),
                },
                'total_runs': len(results),
            }
        
        return stats_by_bin
    
    def print_summary(self) -> None:
        """Print formatted summary of all bin widths"""
        stats_all = self.get_statistics_all_bins()
        
        print("=" * 80)
        print("MULTI-BIN-WIDTH MONTE CARLO SUMMARY")
        print("=" * 80)
        print(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Configuration:")
        print(f"  Total runs: {self.total_runs}")
        print(f"  Photons per run: {self.simulation_config['num_photons']:,}")
        print(f"  Bayesian Blocks p0: {self.simulation_config['p0']}")
        print(f"  Flux steps defined: {self.simulation_config['num_flux_steps']}\n")
        
        print("Bayesian Blocks Detected by Bin Width:")
        print("─" * 80)
        print(f"{'Bin Width':>12} {'Mean':>10} {'Median':>10} {'Std':>10} {'Range':>20} {'95% CI':>20}")
        print("─" * 80)
        
        for bin_width in self.bin_widths:
            if bin_width not in stats_all:
                continue
            
            stats = stats_all[bin_width]['num_blocks']
            mean_val = stats['mean']
            median_val = stats['median']
            std_val = stats['std']
            range_str = f"[{stats['min']}, {stats['max']}]"
            ci_str = f"({stats['95_ci'][0]:.1f}, {stats['95_ci'][1]:.1f})"
            
            print(f"{bin_width:>10.0f}d {mean_val:>10.2f} {median_val:>10.0f} {std_val:>10.2f} {range_str:>20} {ci_str:>20}")
        
        print("─" * 80)
        
        print("\nTime Cells by Bin Width:")
        print("─" * 80)
        print(f"{'Bin Width':>12} {'Mean':>10} {'Std':>10}")
        print("─" * 80)
        
        for bin_width in self.bin_widths:
            if bin_width not in stats_all:
                continue
            
            stats = stats_all[bin_width]['num_cells']
            print(f"{bin_width:>10.0f}d {stats['mean']:>10.0f} {stats['std']:>10.1f}")
        
        print("─" * 80)
        
        print("\nCompression Ratio (cells/blocks) by Bin Width:")
        print("─" * 80)
        print(f"{'Bin Width':>12} {'Mean':>10} {'Std':>10}")
        print("─" * 80)
        
        for bin_width in self.bin_widths:
            if bin_width not in stats_all:
                continue
            
            stats = stats_all[bin_width]['compression_ratio']
            print(f"{bin_width:>10.0f}d {stats['mean']:>10.2f}x {stats['std']:>10.2f}")
        
        print("=" * 80 + "\n")
    
    def plot_comparison(self, figsize: Tuple[float, float] = (16, 12)) -> plt.Figure:
        """
        Create comprehensive comparison across bin widths.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        stats_all = self.get_statistics_all_bins()
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # Collect data for plotting
        bin_labels = [f"{int(bw)}d" for bw in self.bin_widths]
        
        blocks_means = []
        blocks_stds = []
        cells_means = []
        compression_means = []
        
        for bw in self.bin_widths:
            if bw in stats_all:
                blocks_means.append(stats_all[bw]['num_blocks']['mean'])
                blocks_stds.append(stats_all[bw]['num_blocks']['std'])
                cells_means.append(stats_all[bw]['num_cells']['mean'])
                compression_means.append(stats_all[bw]['compression_ratio']['mean'])
        
        # 1. Blocks detected vs. bin width
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.errorbar(self.bin_widths, blocks_means, yerr=blocks_stds, 
                    fmt='o-', capsize=5, markersize=8, linewidth=2, color='steelblue')
        ax1.set_xlabel('Bin Width (days)')
        ax1.set_ylabel('Number of Blocks')
        ax1.set_title('Bayesian Blocks Detected vs. Bin Width')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cells created vs. bin width
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.bin_widths, cells_means, 'o-', markersize=8, linewidth=2, color='coral')
        ax2.set_xlabel('Bin Width (days)')
        ax2.set_ylabel('Number of Cells')
        ax2.set_title('Time Cells Created vs. Bin Width')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Compression ratio vs. bin width
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.bin_widths, compression_means, 'o-', markersize=8, linewidth=2, color='mediumseagreen')
        ax3.set_xlabel('Bin Width (days)')
        ax3.set_ylabel('Compression Ratio (cells/blocks)')
        ax3.set_title('Compression Efficiency vs. Bin Width')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4-6. Distribution plots for each bin width
        for idx, bw in enumerate(self.bin_widths):
            ax = fig.add_subplot(gs[1 + idx//2, idx%2 + 1])
            results = self.results_by_bin[bw]
            blocks = [r.num_blocks for r in results]
            
            ax.hist(blocks, bins=15, color=f'C{idx}', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(blocks), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(blocks):.2f}')
            ax.set_xlabel('Number of Blocks')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Block Distribution ({bw}d bins)')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 7. All distributions overlaid
        ax7 = fig.add_subplot(gs[2, 2])
        for idx, bw in enumerate(self.bin_widths):
            results = self.results_by_bin[bw]
            blocks = np.array([r.num_blocks for r in results])
            ax7.hist(blocks, bins=15, alpha=0.5, label=f"{int(bw)}d", edgecolor='black')
        
        ax7.set_xlabel('Number of Blocks')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Block Distribution Overlay')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(
            f'Multi-Bin-Width Comparison: {self.num_runs_per_bin} runs per bin width\n'
            f'Bayesian Blocks Detection Across Temporal Resolutions',
            fontsize=14, fontweight='bold', y=0.995
        )
        
        return fig
    
    def save_results(self, output_dir: str = './monte_carlo_multi_bin_results') -> None:
        """
        Save all results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print("Saving results...")
        
        # Progress bar for saving
        save_tasks = ['Configuration', 'Statistics', 'Results by bin', 'CSV exports']
        pbar = tqdm(total=len(save_tasks) + len(self.bin_widths), 
                   desc="Saving files", unit="file", ncols=60)
        
        try:
            # Save configuration
            config_file = output_path / 'config.json'
            with open(config_file, 'w') as f:
                json.dump(self.simulation_config, f, indent=2)
            pbar.update(1)
            
            # Save statistics
            stats_all = self.get_statistics_all_bins()
            stats_file = output_path / 'statistics.json'
            with open(stats_file, 'w') as f:
                json.dump(stats_all, f, indent=2, default=str)
            pbar.update(1)
            
            # Save results for each bin width
            for bin_width in self.bin_widths:
                results_data = [r.to_dict() for r in self.results_by_bin[bin_width]]
                bin_file = output_path / f'results_bin_{bin_width:.0f}d.json'
                with open(bin_file, 'w') as f:
                    json.dump(results_data, f, indent=2)
                pbar.update(1)
            
            pbar.update(1)  # Results by bin done
            
            # Save as CSV for each bin width
            for bin_width in self.bin_widths:
                df = pd.DataFrame([r.to_dict() for r in self.results_by_bin[bin_width]])
                csv_file = output_path / f'results_bin_{bin_width:.0f}d.csv'
                df.to_csv(csv_file, index=False)
            
            pbar.close()
            
            if self.verbose:
                print(f"\n✓ Results saved to {output_path}/")
                print(f"  • config.json: Simulation configuration")
                print(f"  • statistics.json: Summary statistics for all bins")
                print(f"  • results_bin_*.json: Individual run results per bin width")
                print(f"  • results_bin_*.csv: Results in CSV format per bin width\n")
        
        except Exception as e:
            pbar.close()
            print(f"Error saving results: {str(e)}")
    
    def to_dataframe_all_bins(self) -> pd.DataFrame:
        """Convert all results to a single DataFrame"""
        all_data = []
        for bw in self.bin_widths:
            for result in self.results_by_bin[bw]:
                all_data.append(result.to_dict())
        
        return pd.DataFrame(all_data)


# Example usage
if __name__ == "__main__":
    
    # Create multi-bin-width Monte Carlo simulation
    mc = MultiBindWidthMonteCarloSimulation(
        bin_widths=[7, 30, 180, 360],
        num_runs_per_bin=50,
        verbose=True
    )
    
    # Define AGN flux profile (3 distinct steps)
    flux_profile = [
        {'start': 54683, 'end': 55500, 'rate': 1.0e-7, 'name': 'Low State'},
        {'start': 55500, 'end': 55700, 'rate': 3.5e-7, 'name': 'High State'},
        {'start': 55700, 'end': 57500, 'rate': 1.5e-7, 'name': 'Intermediate'},
    ]
    
    # Run simulations across all bin widths
    print("\n" + "="*70)
    print("STARTING MULTI-BIN-WIDTH AGN STEPS MONTE CARLO")
    print("="*70 + "\n")
    
    results = mc.run_all_simulations(
        num_photons=40000,
        p0=0.05,
        flux_steps=flux_profile,
        mission_duration=5475
    )
    
    # Print summary statistics
    mc.print_summary()
    
    # Create visualization
    print("Creating visualization...")
    pbar = tqdm(total=2, desc="Generating plots", unit="plot", ncols=60)
    
    fig = mc.plot_comparison(figsize=(16, 12))
    pbar.update(1)
    
    plt.tight_layout()
    pbar.set_description("Saving plot to file")
    plt.savefig('monte_carlo_multi_bin_results.png', dpi=150, bbox_inches='tight')
    pbar.update(1)
    pbar.close()
    
    print("✓ Saved to monte_carlo_multi_bin_results.png\n")
    
    # Save results to files
    mc.save_results('./monte_carlo_multi_bin_results')
    
    # Display results as DataFrame
    print("Results Summary (First 10 rows):")
    df = mc.to_dataframe_all_bins()
    print(df.head(10))
    print(f"\nTotal rows: {len(df)}")
    print("\nBy bin width:")
    print(df.groupby('bin_width')[['num_blocks', 'num_cells', 'compression_ratio']].describe())
    print()
    
    plt.show()