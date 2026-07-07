"""
Monte Carlo Summary Statistics of Bayesian Blocks from source_info_v3
Analyzes the statistical distribution of Bayesian Block detection across all sources
in the agnsteps database (v3). Provides comprehensive summary statistics and distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import seaborn as sns
from tqdm import tqdm
import sys
from scipy import stats as scipy_stats

# Configure styling
plt.style.use('dark_background')
sns.set_theme('talk', font_scale=1.0)


@dataclass
class SourceBlockStatistics:
    """Statistics for a single source"""
    name: str
    num_blocks: int
    num_photons: int
    ts: float
    variability: float
    association: str
    eflux100: float
    

class BayesianBlocksV3Statistics:
    """
    Calculate comprehensive statistics on Bayesian Blocks detection
    from the complete source_info_v3 database.
    """
    
    def __init__(self, 
                 info_file: str = 'files/source_info_v3.pkl',
                 verbose: bool = True):
        """
        Initialize statistics calculator.
        
        Args:
            info_file: Path to source_info_v3 pickle file
            verbose: Print progress information
        """
        self.info_file = Path(info_file)
        self.verbose = verbose
        self.vdb = None
        self.source_stats = []
        self.summary_stats = {}
        self.timestamp = pd.Timestamp.now()
        
    def load_database(self) -> bool:
        """
        Load the source_info_v3 database.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.info_file.exists():
                print(f"Error: {self.info_file} not found")
                return False
            
            with open(self.info_file, 'rb') as f:
                self.vdb = pickle.load(f)
            
            if self.verbose:
                print(f"✓ Loaded source_info_v3 database")
                print(f"  Total sources: {len(self.vdb)}\n")
            
            return True
            
        except Exception as e:
            print(f"Error loading database: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_source_info(self, source_name: str, source_dict: Dict) -> Optional[SourceBlockStatistics]:
        """
        Extract key information from a source.
        
        Args:
            source_name: Name of the source
            source_dict: Source data dictionary
            
        Returns:
            SourceBlockStatistics object
        """
        try:
            # Get light curve data
            lc = source_dict.get('light_curve')
            
            # Count blocks
            if lc is None:
                num_blocks = 0
            elif isinstance(lc, dict):
                # Handle dict format (v1/v3)
                num_blocks = len(lc.get('tw', []))
            else:
                # Handle DataFrame format
                num_blocks = len(lc)
            
            # Get other metadata if available
            num_photons = source_dict.get('num_photons', 0)
            ts = source_dict.get('ts', 0)
            variability = source_dict.get('variability', 0)
            association = source_dict.get('association', 'unknown')
            eflux100 = source_dict.get('eflux100', 0)
            
            return SourceBlockStatistics(
                name=source_name,
                num_blocks=num_blocks,
                num_photons=num_photons,
                ts=ts,
                variability=variability,
                association=association,
                eflux100=eflux100,
            )
        
        except Exception as e:
            if self.verbose:
                print(f"Error extracting info for {source_name}: {e}")
            return None
    
    def calculate_all_statistics(self) -> Dict:
        """
        Calculate comprehensive statistics on Bayesian Blocks.
        
        Returns:
            Dictionary with all calculated statistics
        """
        if self.vdb is None:
            print("Error: Database not loaded")
            return {}
        
        if self.verbose:
            print("="*80)
            print("CALCULATING BAYESIAN BLOCKS STATISTICS")
            print("="*80 + "\n")
            print("Extracting source information...")
        
        self.source_stats = []
        
        # Extract stats for all sources
        pbar = tqdm(total=len(self.vdb), desc="Processing sources", 
                   unit="source", ncols=80, disable=not self.verbose)
        
        for source_name, source_dict in self.vdb.items():
            stats = self.extract_source_info(source_name, source_dict)
            if stats:
                self.source_stats.append(stats)
            pbar.update(1)
        
        pbar.close()
        
        if self.verbose:
            print(f"\n✓ Extracted statistics for {len(self.source_stats)} sources\n")
        
        # Calculate statistics
        if not self.source_stats:
            return {}
        
        num_blocks_all = np.array([s.num_blocks for s in self.source_stats])
        
        # Filter sources with blocks > 0
        sources_with_blocks = [s for s in self.source_stats if s.num_blocks > 0]
        num_blocks_nonzero = np.array([s.num_blocks for s in sources_with_blocks])
        
        # Group by association type
        associations = {}
        for s in self.source_stats:
            if s.association not in associations:
                associations[s.association] = []
            associations[s.association].append(s.num_blocks)
        
        # Group by block count
        by_block_count = {}
        for s in self.source_stats:
            if s.num_blocks not in by_block_count:
                by_block_count[s.num_blocks] = 0
            by_block_count[s.num_blocks] += 1
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = {p: float(np.percentile(num_blocks_nonzero, p)) for p in percentiles}
        
        # Statistics on sources with blocks
        stats_dict = {
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'total_sources': len(self.source_stats),
            'sources_with_blocks': len(sources_with_blocks),
            'sources_without_blocks': len(self.source_stats) - len(sources_with_blocks),
            'percentage_with_blocks': len(sources_with_blocks) / len(self.source_stats) * 100,
            
            # All sources
            'all_sources': {
                'mean_blocks': float(np.mean(num_blocks_all)),
                'median_blocks': float(np.median(num_blocks_all)),
                'std_blocks': float(np.std(num_blocks_all)),
                'min_blocks': int(np.min(num_blocks_all)),
                'max_blocks': int(np.max(num_blocks_all)),
                'q1': float(np.percentile(num_blocks_all, 25)),
                'q3': float(np.percentile(num_blocks_all, 75)),
            },
            
            # Non-zero sources
            'nonzero_sources': {
                'mean_blocks': float(np.mean(num_blocks_nonzero)),
                'median_blocks': float(np.median(num_blocks_nonzero)),
                'std_blocks': float(np.std(num_blocks_nonzero)),
                'min_blocks': int(np.min(num_blocks_nonzero)),
                'max_blocks': int(np.max(num_blocks_nonzero)),
                'q1': float(np.percentile(num_blocks_nonzero, 25)),
                'q3': float(np.percentile(num_blocks_nonzero, 75)),
                'skewness': float(scipy_stats.skew(num_blocks_nonzero)),
                'kurtosis': float(scipy_stats.kurtosis(num_blocks_nonzero)),
            },
            
            'percentiles': percentile_values,
            'by_block_count': by_block_count,
            'by_association': {},
        }
        
        # Statistics by association
        for assoc, blocks in associations.items():
            blocks_arr = np.array(blocks)
            blocks_nonzero = blocks_arr[blocks_arr > 0]
            
            stats_dict['by_association'][assoc] = {
                'total_count': len(blocks),
                'with_blocks_count': len(blocks_nonzero),
                'mean_blocks': float(np.mean(blocks_arr)),
                'median_blocks': float(np.median(blocks_arr)),
                'std_blocks': float(np.std(blocks_arr)),
                'min_blocks': int(np.min(blocks_arr)),
                'max_blocks': int(np.max(blocks_arr)),
                'percentage_with_blocks': len(blocks_nonzero) / len(blocks) * 100 if blocks else 0,
            }
        
        self.summary_stats = stats_dict
        return stats_dict
    
    def print_summary_statistics(self) -> None:
        """Print formatted summary statistics."""
        if not self.summary_stats:
            print("Error: Statistics not calculated")
            return
        
        stats = self.summary_stats
        
        print("\n" + "="*80)
        print("BAYESIAN BLOCKS STATISTICAL SUMMARY - source_info_v3")
        print("="*80 + "\n")
        
        print(f"Timestamp: {stats['timestamp']}\n")
        
        print(f"Database Overview:")
        print(f"  Total sources: {stats['total_sources']:,}")
        print(f"  Sources with blocks: {stats['sources_with_blocks']:,} ({stats['percentage_with_blocks']:.1f}%)")
        print(f"  Sources without blocks: {stats['sources_without_blocks']:,}\n")
        
        print(f"All Sources (including those with 0 blocks):")
        all_stats = stats['all_sources']
        print(f"  Mean: {all_stats['mean_blocks']:.3f}")
        print(f"  Median: {all_stats['median_blocks']:.1f}")
        print(f"  Std Dev: {all_stats['std_blocks']:.3f}")
        print(f"  Min: {all_stats['min_blocks']}")
        print(f"  Max: {all_stats['max_blocks']}")
        print(f"  Q1 (25%): {all_stats['q1']:.1f}")
        print(f"  Q3 (75%): {all_stats['q3']:.1f}\n")
        
        print(f"Sources with ≥1 Blocks:")
        nz_stats = stats['nonzero_sources']
        print(f"  Mean: {nz_stats['mean_blocks']:.3f}")
        print(f"  Median: {nz_stats['median_blocks']:.1f}")
        print(f"  Std Dev: {nz_stats['std_blocks']:.3f}")
        print(f"  Min: {nz_stats['min_blocks']}")
        print(f"  Max: {nz_stats['max_blocks']}")
        print(f"  Q1 (25%): {nz_stats['q1']:.1f}")
        print(f"  Q3 (75%): {nz_stats['q3']:.1f}")
        print(f"  Skewness: {nz_stats['skewness']:.3f}")
        print(f"  Kurtosis: {nz_stats['kurtosis']:.3f}\n")
        
        print(f"Block Count Distribution:")
        print(f"  {'Blocks':>8} {'Count':>8} {'Percentage':>12}")
        print(f"  {'-'*8} {'-'*8} {'-'*12}")
        
        for block_count in sorted(stats['by_block_count'].keys()):
            count = stats['by_block_count'][block_count]
            percentage = (count / stats['total_sources']) * 100
            print(f"  {block_count:>8} {count:>8} {percentage:>11.2f}%")
        
        print(f"\nPercentile Distribution (non-zero sources):")
        print(f"  {'Percentile':>12} {'Value':>12}")
        print(f"  {'-'*12} {'-'*12}")
        
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            val = stats['percentiles'].get(p, 0)
            print(f"  {p:>11}% {val:>12.1f}")
        
        print(f"\nStatistics by Source Association:")
        print(f"  {'Association':>15} {'Total':>8} {'With BB':>8} {'%':>8} {'Mean':>8} {'Median':>8}")
        print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        for assoc in sorted(stats['by_association'].keys()):
            assoc_stats = stats['by_association'][assoc]
            print(f"  {assoc:>15} {assoc_stats['total_count']:>8} {assoc_stats['with_blocks_count']:>8} "
                  f"{assoc_stats['percentage_with_blocks']:>7.1f}% {assoc_stats['mean_blocks']:>8.2f} "
                  f"{assoc_stats['median_blocks']:>8.1f}")
        
        print("\n" + "="*80 + "\n")
    
    def plot_statistics(self, figsize: Tuple[float, float] = (16, 12)) -> plt.Figure:
        """
        Create comprehensive visualization of block statistics.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        if not self.source_stats:
            print("Error: No statistics calculated")
            return None
        
        stats = self.summary_stats
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        num_blocks_all = np.array([s.num_blocks for s in self.source_stats])
        sources_with_blocks = [s for s in self.source_stats if s.num_blocks > 0]
        num_blocks_nonzero = np.array([s.num_blocks for s in sources_with_blocks])
        
        # 1. Histogram of all blocks
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(num_blocks_all, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(num_blocks_all), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(num_blocks_all):.2f}')
        ax1.axvline(np.median(num_blocks_all), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(num_blocks_all):.1f}')
        ax1.set_xlabel('Number of Blocks', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Histogram: All Sources', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Histogram of non-zero blocks
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(num_blocks_nonzero, bins=40, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(num_blocks_nonzero), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(num_blocks_nonzero):.2f}')
        ax2.axvline(np.median(num_blocks_nonzero), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(num_blocks_nonzero):.1f}')
        ax2.set_xlabel('Number of Blocks', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Histogram: Sources with Blocks', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Block count distribution (bar chart)
        ax3 = fig.add_subplot(gs[0, 2])
        block_counts = sorted(stats['by_block_count'].keys())
        counts = [stats['by_block_count'][b] for b in block_counts]
        ax3.bar(block_counts, counts, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Number of Blocks', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Number of Sources', fontsize=11, fontweight='bold')
        ax3.set_title('Block Count Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Box plot
        ax4 = fig.add_subplot(gs[1, 0])
        bp = ax4.boxplot([num_blocks_all, num_blocks_nonzero], 
                         labels=['All Sources', 'Non-Zero'],
                         patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax4.set_ylabel('Number of Blocks', fontsize=11, fontweight='bold')
        ax4.set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. CDF plot
        ax5 = fig.add_subplot(gs[1, 1])
        sorted_nonzero = np.sort(num_blocks_nonzero)
        cdf = np.arange(1, len(sorted_nonzero) + 1) / len(sorted_nonzero)
        ax5.plot(sorted_nonzero, cdf, 'o-', markersize=4, alpha=0.7, linewidth=2)
        ax5.set_xlabel('Number of Blocks', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
        ax5.set_title('CDF: Non-Zero Sources', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Percentile plot
        ax6 = fig.add_subplot(gs[1, 2])
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_vals = [stats['percentiles'].get(p, 0) for p in percentiles]
        ax6.plot(percentiles, percentile_vals, 'o-', markersize=8, linewidth=2, color='steelblue')
        ax6.fill_between(percentiles, percentile_vals, alpha=0.3)
        ax6.set_xlabel('Percentile', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Number of Blocks', fontsize=11, fontweight='bold')
        ax6.set_title('Percentile Distribution', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. By association - bar chart
        ax7 = fig.add_subplot(gs[2, 0])
        assocs = sorted(stats['by_association'].keys())
        means = [stats['by_association'][a]['mean_blocks'] for a in assocs]
        ax7.bar(range(len(assocs)), means, color='lightgreen', edgecolor='black', alpha=0.7)
        ax7.set_xticks(range(len(assocs)))
        ax7.set_xticklabels(assocs, rotation=45, ha='right')
        ax7.set_ylabel('Mean Number of Blocks', fontsize=11, fontweight='bold')
        ax7.set_title('Mean Blocks by Association', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Percentage with blocks by association
        ax8 = fig.add_subplot(gs[2, 1])
        percentages = [stats['by_association'][a]['percentage_with_blocks'] for a in assocs]
        colors_pct = ['green' if p > 50 else 'orange' if p > 20 else 'red' for p in percentages]
        ax8.bar(range(len(assocs)), percentages, color=colors_pct, edgecolor='black', alpha=0.7)
        ax8.set_xticks(range(len(assocs)))
        ax8.set_xticklabels(assocs, rotation=45, ha='right')
        ax8.set_ylabel('Percentage with Blocks (%)', fontsize=11, fontweight='bold')
        ax8.set_title('Block Detection Rate by Association', fontsize=12, fontweight='bold')
        ax8.set_ylim(0, 100)
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. Summary statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        summary_text = f"""
SUMMARY STATISTICS

Total Sources: {stats['total_sources']:,}
With Blocks: {stats['sources_with_blocks']:,}
({stats['percentage_with_blocks']:.1f}%)

All Sources:
  Mean: {stats['all_sources']['mean_blocks']:.2f}
  Median: {stats['all_sources']['median_blocks']:.0f}
  Std: {stats['all_sources']['std_blocks']:.2f}

With Blocks:
  Mean: {stats['nonzero_sources']['mean_blocks']:.2f}
  Median: {stats['nonzero_sources']['median_blocks']:.0f}
  Std: {stats['nonzero_sources']['std_blocks']:.2f}
  
  Skewness: {stats['nonzero_sources']['skewness']:.3f}
  Kurtosis: {stats['nonzero_sources']['kurtosis']:.3f}
"""
        
        ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
                fontfamily='monospace', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, linewidth=2))
        
        fig.suptitle(
            f'Bayesian Blocks Statistical Analysis - source_info_v3\n'
            f'{stats["total_sources"]:,} Sources Analyzed',
            fontsize=14, fontweight='bold', y=0.995
        )
        
        return fig
    
    def save_statistics(self, output_dir: str = './bb_v3_statistics') -> None:
        """Save statistics to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print("Saving statistics...")
        
        try:
            # Save summary statistics as JSON
            import json
            stats_file = output_path / 'summary_statistics.json'
            with open(stats_file, 'w') as f:
                json.dump(self.summary_stats, f, indent=2)
            
            # Save source list as CSV
            df_sources = pd.DataFrame([
                {
                    'name': s.name,
                    'num_blocks': s.num_blocks,
                    'num_photons': s.num_photons,
                    'ts': s.ts,
                    'variability': s.variability,
                    'association': s.association,
                    'eflux100': s.eflux100,
                }
                for s in self.source_stats
            ])
            
            csv_file = output_path / 'source_blocks.csv'
            df_sources.to_csv(csv_file, index=False)
            
            # Save block count distribution
            block_dist_file = output_path / 'block_distribution.txt'
            with open(block_dist_file, 'w') as f:
                f.write("Block Count Distribution\n")
                f.write("="*40 + "\n")
                for block_count in sorted(self.summary_stats['by_block_count'].keys()):
                    count = self.summary_stats['by_block_count'][block_count]
                    percentage = (count / self.summary_stats['total_sources']) * 100
                    f.write(f"{block_count:3d} blocks: {count:6d} sources ({percentage:6.2f}%)\n")
            
            if self.verbose:
                print(f"\n✓ Statistics saved to {output_path}/")
                print(f"  • summary_statistics.json: Full statistics")
                print(f"  • source_blocks.csv: Source-by-source data")
                print(f"  • block_distribution.txt: Block count breakdown\n")
        
        except Exception as e:
            print(f"Error saving statistics: {e}")


# Main execution
if __name__ == "__main__":
    
    # Create statistics calculator
    calculator = BayesianBlocksV3Statistics(
        info_file='files/source_info_v3.pkl',
        verbose=True
    )
    
    # Load database
    if calculator.load_database():
        
        # Calculate statistics
        stats = calculator.calculate_all_statistics()
        
        # Print results
        calculator.print_summary_statistics()
        
        # Create visualization
        print("Creating comprehensive visualizations...")
        fig = calculator.plot_statistics(figsize=(16, 12))
        
        if fig is not None:
            plt.savefig('bb_v3_statistics.png', dpi=150, bbox_inches='tight')
            print("✓ Saved plot: bb_v3_statistics.png\n")
            
            # Save statistics
            calculator.save_statistics('./bb_v3_statistics')
            
            plt.show()
        else:
            print("Failed to create visualization")
    
    else:
        print("Failed to load database")