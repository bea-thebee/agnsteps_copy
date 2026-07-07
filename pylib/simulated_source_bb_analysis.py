"""
Monte Carlo simulation for the distribution of Bayesian block counts.

This module provides tools to simulate the number of Bayesian blocks
expected under null hypothesis (no variability) and compare with 
observed distributions.

Useful for:
- Understanding the false positive rate of block detection
- Setting significance thresholds for variability claims
- Validating the BB algorithm performance
- Testing hypothesis about intrinsic source variability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import seaborn as sns
from scipy import stats
from pylib import simulated_source_data

from simulated_source_data import (
    SimulatedSourceDatabase,
    LightCurveParameters,
    FluxParameters,
    SimulatedLightCurve,
    SourcePropertyGenerator
)


@dataclass
class BBCountSimulationConfig:
    """Configuration for Bayesian block count simulations"""
    n_simulations: int = 1000      # Number of MC realizations
    n_sources: int = 100           # Sources per realization
    n_bins_per_lc: int = 100       # Time bins per light curve
    mean_bin_width: float = 30     # Days
    
    # Null hypothesis: constant flux with Poisson noise
    use_constant_flux: bool = True
    constant_flux_level: float = 1.0
    
    # Block generation parameters
    prob_block_per_bin: float = 0.03  # P(new block at any bin)
    
    # Seed for reproducibility
    seed_base: int = 0


class BayesianBlockCounter:
    """Simple stochastic BB counter for simulation"""
    
    @staticmethod
    def count_blocks_stochastic(n_bins: int, 
                               prob_new_block: float,
                               random_state: np.random.RandomState = None) -> int:
        """
        Count expected number of blocks using stochastic process
        
        Parameters
        ----------
        n_bins : int
            Number of time bins
        prob_new_block : float
            Probability of new block boundary at each bin
        random_state : np.random.RandomState
            Random number generator
        
        Returns
        -------
        n_blocks : int
            Number of blocks generated
        """
        if random_state is None:
            random_state = np.random.RandomState()
        
        # Start with first block
        blocks = [0]
        
        # Stochastically add block boundaries
        for i in range(1, n_bins):
            if random_state.random() < prob_new_block:
                blocks.append(i)
        
        # Always include last bin
        if blocks[-1] != n_bins - 1:
            blocks.append(n_bins - 1)
        
        return len(blocks)
    
    @staticmethod
    def count_blocks_from_lightcurve(flux: np.ndarray,
                                     errors: Optional[np.ndarray] = None,
                                     p0: float = 0.05) -> int:
        """
        Count blocks in a light curve using simplified BB algorithm
        
        This is a simplified version that counts significant flux changes
        
        Parameters
        ----------
        flux : np.ndarray
            Flux values
        errors : np.ndarray, optional
            Flux errors
        p0 : float
            Significance threshold for new block
        
        Returns
        -------
        n_blocks : int
            Number of detected blocks
        """
        if errors is None:
            errors = np.ones_like(flux) * 0.1
        
        # Compute significance of flux changes
        n_bins = len(flux)
        blocks = [0]
        
        for i in range(1, n_bins):
            # Z-score for flux change
            flux_change = abs(flux[i] - flux[i-1])
            combined_error = np.sqrt(errors[i]**2 + errors[i-1]**2)
            
            if combined_error > 0:
                z_score = flux_change / combined_error
                
                # New block if change is significant
                if z_score > stats.norm.ppf(1 - p0/2):
                    blocks.append(i)
        
        # Include last bin
        if blocks[-1] != n_bins - 1:
            blocks.append(n_bins - 1)
        
        return len(blocks)


class MonteCarloBlockSimulation:
    """Run Monte Carlo simulations of Bayesian block counts"""
    
    def __init__(self, config: BBCountSimulationConfig = None):
        """
        Initialize the simulation
        
        Parameters
        ----------
        config : BBCountSimulationConfig
            Simulation configuration
        """
        self.config = config or BBCountSimulationConfig()
        self.results = []
        self.detailed_results = None
    
    def run_null_hypothesis_simulation(self) -> pd.DataFrame:
        """
        Run simulations under null hypothesis (constant flux, no variability)
        
        Returns
        -------
        df_results : pd.DataFrame
            Results with columns: simulation_id, source_id, n_blocks, flux_level
        """
        print(f"Running {self.config.n_simulations} MC simulations...")
        print(f"  Sources per sim: {self.config.n_sources}")
        print(f"  Bins per source: {self.config.n_bins_per_lc}")
        print(f"  P(block): {self.config.prob_block_per_bin:.4f}")
        
        results = []
        
        for sim_id in range(self.config.n_simulations):
            seed = self.config.seed_base + sim_id
            rng = np.random.RandomState(seed)
            
            # Generate sources under null hypothesis
            for source_id in range(self.config.n_sources):
                
                # Generate light curve with constant flux
                if self.config.use_constant_flux:
                    # Constant flux with Poisson noise
                    flux = rng.poisson(
                        self.config.constant_flux_level,
                        self.config.n_bins_per_lc
                    ) / self.config.constant_flux_level
                    
                    # Errors from Poisson statistics
                    errors = np.sqrt(flux) / self.config.constant_flux_level
                
                else:
                    # Random constant level (more realistic)
                    level = rng.uniform(0.5, 2.0)
                    flux = rng.normal(
                        level,
                        level * 0.1,
                        self.config.n_bins_per_lc
                    )
                    flux = np.clip(flux, 0.01, 10)
                    errors = np.abs(flux) * 0.1
                
                # Count blocks
                n_blocks = BayesianBlockCounter.count_blocks_from_lightcurve(
                    flux, errors
                )
                
                results.append({
                    'simulation_id': sim_id,
                    'source_id': source_id,
                    'n_blocks': n_blocks,
                    'flux_level': np.mean(flux),
                    'flux_std': np.std(flux),
                })
            
            # Progress update
            if (sim_id + 1) % max(1, self.config.n_simulations // 10) == 0:
                print(f"  Completed {sim_id + 1}/{self.config.n_simulations}")
        
        self.detailed_results = pd.DataFrame(results)
        return self.detailed_results
    
    def run_variable_source_simulation(self,
                                       n_steps: List[int] = None,
                                       step_sizes: List[float] = None) -> pd.DataFrame:
        """
        Run simulations with injected variability (steps/blocks)
        
        Parameters
        ----------
        n_steps : List[int]
            Number of steps to inject (e.g., [1, 2, 3])
        step_sizes : List[float]
            Flux ratios for steps (e.g., [1.5, 2.0])
        
        Returns
        -------
        df_results : pd.DataFrame
            Results with columns: simulation_id, source_id, n_injected_blocks, 
            n_detected_blocks, n_bins_per_block
        """
        if n_steps is None:
            n_steps = [1, 2, 3]
        if step_sizes is None:
            step_sizes = [1.5, 2.0]
        
        print(f"Running simulations with injected variability...")
        print(f"  Injected blocks: {n_steps}")
        print(f"  Step sizes: {step_sizes}")
        
        results = []
        
        for sim_id in range(self.config.n_simulations):
            seed = self.config.seed_base + sim_id
            rng = np.random.RandomState(seed)
            
            for source_id, n_inject in enumerate(n_steps * self.config.n_sources // len(n_steps)):
                for step_size in step_sizes:
                    
                    # Create light curve with injected steps
                    n_bins = self.config.n_bins_per_lc
                    bins_per_block = n_bins // (n_inject + 1)
                    
                    flux = np.ones(n_bins)
                    current_level = 1.0
                    
                    # Inject steps
                    for step_idx in range(n_inject):
                        start = (step_idx + 1) * bins_per_block
                        current_level *= step_size
                        flux[start:] *= current_level
                    
                    # Add noise
                    flux = flux * rng.normal(1.0, 0.1, n_bins)
                    flux = np.clip(flux, 0.01, 10)
                    errors = np.abs(flux) * 0.1
                    
                    # Count detected blocks
                    n_detected = BayesianBlockCounter.count_blocks_from_lightcurve(
                        flux, errors
                    )
                    
                    results.append({
                        'simulation_id': sim_id,
                        'source_id': source_id,
                        'step_size': step_size,
                        'n_injected_blocks': n_inject + 1,
                        'n_detected_blocks': n_detected,
                        'detection_efficiency': 1.0 if n_detected == n_inject + 1 else 0.0,
                        'n_bins_per_block': bins_per_block,
                        'mean_flux_ratio': np.max(flux) / np.min(flux),
                    })
            
            if (sim_id + 1) % max(1, self.config.n_simulations // 10) == 0:
                print(f"  Completed {sim_id + 1}/{self.config.n_simulations}")
        
        self.detailed_results = pd.DataFrame(results)
        return self.detailed_results
    
    def summarize_results(self) -> Dict:
        """
        Generate summary statistics from simulation results
        
        Returns
        -------
        summary : Dict
            Summary statistics
        """
        if self.detailed_results is None:
            raise ValueError("No results to summarize. Run a simulation first.")
        
        df = self.detailed_results
        
        summary = {
            'n_total_sources': len(df),
            'n_simulations': df['simulation_id'].nunique(),
            'sources_per_sim': len(df) // df['simulation_id'].nunique(),
        }
        
        if 'n_blocks' in df.columns:
            # Null hypothesis results
            summary.update({
                'mean_n_blocks': df['n_blocks'].mean(),
                'median_n_blocks': df['n_blocks'].median(),
                'std_n_blocks': df['n_blocks'].std(),
                'min_n_blocks': df['n_blocks'].min(),
                'max_n_blocks': df['n_blocks'].max(),
                'frac_single_block': (df['n_blocks'] == 1).sum() / len(df),
                'frac_two_blocks': (df['n_blocks'] == 2).sum() / len(df),
                'frac_three_plus_blocks': (df['n_blocks'] >= 3).sum() / len(df),
            })
            
            # Block distribution
            summary['block_distribution'] = df['n_blocks'].value_counts().to_dict()
        
        if 'n_detected_blocks' in df.columns:
            # Injected variability results
            summary.update({
                'mean_detection_efficiency': df['detection_efficiency'].mean(),
                'detection_efficiency_by_step': df.groupby('step_size')['detection_efficiency'].mean().to_dict(),
            })
        
        return summary


class BlockCountVisualizer:
    """Create visualizations of BB count distributions"""
    
    @staticmethod
    def plot_null_hypothesis_distribution(sim: MonteCarloBlockSimulation,
                                         figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot distribution of block counts under null hypothesis
        
        Parameters
        ----------
        sim : MonteCarloBlockSimulation
            Completed simulation
        figsize : Tuple[int, int]
            Figure size
        
        Returns
        -------
        fig : plt.Figure
            Matplotlib figure
        """
        df = sim.detailed_results
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Histogram of block counts
        ax = axes[0, 0]
        ax.hist(df['n_blocks'], bins=np.arange(0.5, df['n_blocks'].max() + 1.5),
               histtype='step', color='cyan', lw=2, edgecolor='black')
        ax.set_xlabel('Number of Bayesian Blocks')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Block Counts (Null Hypothesis)')
        ax.grid(alpha=0.3)
        
        # Plot 2: Block count by simulation
        ax = axes[0, 1]
        sim_means = df.groupby('simulation_id')['n_blocks'].mean()
        ax.hist(sim_means, bins=20, histtype='step', color='magenta', lw=2, edgecolor='black')
        ax.set_xlabel('Mean Blocks per Simulation')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Mean Blocks Across Simulations')
        ax.grid(alpha=0.3)
        
        # Plot 3: CDF of block counts
        ax = axes[0, 2]
        sorted_blocks = np.sort(df['n_blocks'].values)
        cdf = np.arange(1, len(sorted_blocks) + 1) / len(sorted_blocks)
        ax.plot(sorted_blocks, cdf, drawstyle='steps-post', lw=2, color='cyan')
        ax.set_xlabel('Number of Bayesian Blocks')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('CDF of Block Counts')
        ax.grid(alpha=0.3)
        
        # Plot 4: Block count distribution by simulation
        ax = axes[1, 0]
        data_by_sim = [df[df['simulation_id'] == sim_id]['n_blocks'].values 
                       for sim_id in sorted(df['simulation_id'].unique())[:5]]
        bp = ax.boxplot(data_by_sim, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('cyan')
        ax.set_xlabel('Simulation ID')
        ax.set_ylabel('Number of Blocks')
        ax.set_title('Block Count Distribution (First 5 Simulations)')
        ax.grid(alpha=0.3, axis='y')
        
        # Plot 5: Mean flux vs n_blocks
        ax = axes[1, 1]
        scatter = ax.scatter(df['flux_level'], df['n_blocks'], alpha=0.3, s=10, c=df['flux_std'])
        ax.set_xlabel('Mean Flux Level')
        ax.set_ylabel('Number of Blocks')
        ax.set_title('Flux Level vs Block Count')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Flux Std Dev')
        ax.grid(alpha=0.3)
        
        # Plot 6: Fraction as function of block count
        ax = axes[1, 2]
        block_counts = df['n_blocks'].value_counts().sort_index()
        total = len(df)
        fractions = block_counts / total
        ax.bar(fractions.index, fractions.values, color='cyan', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Blocks')
        ax.set_ylabel('Fraction of Sources')
        ax.set_title('Fraction Distribution of Block Counts')
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_detection_efficiency(sim: MonteCarloBlockSimulation,
                                 figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Plot detection efficiency for injected variability
        
        Parameters
        ----------
        sim : MonteCarloBlockSimulation
            Completed simulation with injected variability
        figsize : Tuple[int, int]
            Figure size
        
        Returns
        -------
        fig : plt.Figure
            Matplotlib figure
        """
        df = sim.detailed_results
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Detection efficiency vs step size
        ax = axes[0]
        eff_by_step = df.groupby('step_size')['detection_efficiency'].mean()
        n_by_step = df.groupby('step_size').size()
        ax.errorbar(
            eff_by_step.index,
            eff_by_step.values,
            fmt='o-',
            lw=2,
            ms=8,
            color='cyan',
            capsize=5
        )
        ax.set_xlabel('Injected Step Size (Flux Ratio)')
        ax.set_ylabel('Detection Efficiency')
        ax.set_title('Ability to Detect Injected Steps')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(alpha=0.3)
        
        # Plot 2: Detected vs injected blocks
        ax = axes[1]
        scatter = ax.scatter(
            df['n_injected_blocks'],
            df['n_detected_blocks'],
            alpha=0.5,
            s=30,
            c=df['step_size'],
            cmap='viridis'
        )
        
        # Perfect detection line
        max_blocks = max(df['n_injected_blocks'].max(), df['n_detected_blocks'].max())
        ax.plot([0, max_blocks], [0, max_blocks], 'r--', lw=2, label='Perfect detection')
        
        ax.set_xlabel('Injected Blocks')
        ax.set_ylabel('Detected Blocks')
        ax.set_title('Detected vs Injected Block Counts')
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Step Size')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig


def compare_simulations(config_list: List[BBCountSimulationConfig],
                       labels: List[str],
                       output_dir: str = './bb_mc_results') -> Dict:
    """
    Run multiple simulations with different configurations and compare
    
    Parameters
    ----------
    config_list : List[BBCountSimulationConfig]
        List of configurations to test
    labels : List[str]
        Labels for each configuration
    output_dir : str
        Directory to save results
    
    Returns
    -------
    comparison_results : Dict
        Dictionary with results and summaries
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    summaries = {}
    
    for config, label in zip(config_list, labels):
        print(f"\n{'='*60}")
        print(f"Configuration: {label}")
        print(f"{'='*60}")
        
        sim = MonteCarloBlockSimulation(config)
        df_results = sim.run_null_hypothesis_simulation()
        summary = sim.summarize_results()
        
        results[label] = (sim, df_results)
        summaries[label] = summary
        
        # Save results
        df_results.to_csv(f"{output_dir}/results_{label}.csv", index=False)
        
        # Print summary
        print(f"\n{label} Summary:")
        print(f"  Mean blocks: {summary['mean_n_blocks']:.2f} ± {summary['std_n_blocks']:.2f}")
        print(f"  Single block: {summary['frac_single_block']:.1%}")
        print(f"  Two blocks: {summary['frac_two_blocks']:.1%}")
        print(f"  3+ blocks: {summary['frac_three_plus_blocks']:.1%}")
    
    # Create comparison figure
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(labels))
    
    for i, (sim, _) in enumerate(results.values()):
        data = sim.detailed_results['n_blocks']
        bp = ax.boxplot([data], positions=[positions[i]], widths=0.5, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(plt.cm.Set2(i))
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Number of Bayesian Blocks')
    ax.set_title('Comparison of Block Count Distributions')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison.png", dpi=150)
    
    return {
        'results': results,
        'summaries': summaries,
        'comparison_figure': fig,
    }


# Example usage and demonstrations
if __name__ == '__main__':
    print("Monte Carlo Simulation of Bayesian Block Counts")
    print("="*60)
    
    # Example 1: Null hypothesis (no variability)
    print("\nExample 1: Null Hypothesis Simulation")
    config_null = BBCountSimulationConfig(
        n_simulations=10,
        n_sources=100,
        n_bins_per_lc=100,
        use_constant_flux=True,
        prob_block_per_bin=0.03,
        seed_base=42
    )
    
    sim_null = MonteCarloBlockSimulation(config_null)
    df_null = sim_null.run_null_hypothesis_simulation()
    summary_null = sim_null.summarize_results()
    
    print("\nNull Hypothesis Summary:")
    for key, value in summary_null.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Visualize
    fig_null = BlockCountVisualizer.plot_null_hypothesis_distribution(sim_null)
    plt.savefig('./bb_mc_null_hypothesis.png', dpi=150)
    
    # Example 2: Variable sources with injected steps
    print("\n" + "="*60)
    print("Example 2: Injected Variability Simulation")
    config_var = BBCountSimulationConfig(
        n_simulations=5,
        n_sources=50,
        n_bins_per_lc=100,
        seed_base=100
    )
    
    sim_var = MonteCarloBlockSimulation(config_var)
    df_var = sim_var.run_variable_source_simulation(
        n_steps=[1, 2],
        step_sizes=[1.5, 2.0, 3.0]
    )
    
    fig_var = BlockCountVisualizer.plot_detection_efficiency(sim_var)
    plt.savefig('./bb_mc_detection_efficiency.png', dpi=150)
    
    # Example 3: Compare configurations
    print("\n" + "="*60)
    print("Example 3: Configuration Comparison")
    
    configs = [
        BBCountSimulationConfig(n_simulations=5, prob_block_per_bin=0.01, seed_base=200),
        BBCountSimulationConfig(n_simulations=5, prob_block_per_bin=0.05, seed_base=300),
        BBCountSimulationConfig(n_simulations=5, prob_block_per_bin=0.10, seed_base=400),
    ]
    
    labels = ['P(block)=0.01', 'P(block)=0.05', 'P(block)=0.10']
    
    comparison = compare_simulations(configs, labels, output_dir='./bb_mc_comparison')
    
    print("\nComparison complete! Results saved.")
