"""
Detailed Analysis of AGN Steps and Light Curve Variability
Based on agnsteps analysis framework and source_info_v1 database

This module performs comprehensive analysis of a specific AGN source,
examining its Bayesian Block structure, flux transitions, and variability characteristics.
Flexibly handles source name formats.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import seaborn as sns

# Configure styling
plt.style.use('dark_background')
sns.set_theme('talk', font_scale=1.0)


@dataclass
class SourceInfo:
    """Container for source analysis data"""
    name: str
    uw_name: str
    association: str
    ts: float
    bbvar: float
    eflux100: float
    r95: float
    nbb: int
    light_curve: Optional[pd.DataFrame]
    
    
class AGNStepAnalyzer:
    """
    Comprehensive analysis of AGN step structure using agnsteps methodology.
    """
    
    def __init__(self, source_name: str = 'J0725.8-0054', 
                 info_file: str = 'files/source_info_v1.pkl'):
        """
        Initialize analyzer for a specific source.
        
        Args:
            source_name: Name of source to analyze (can be with or without 4FGL prefix)
            info_file: Path to source_info pickle file
        """
        self.source_name_input = source_name
        self.source_name = None  # Will be set after loading
        self.info_file = Path(info_file)
        self.source_data = None
        self.light_curve_df = None
        self.analysis_results = {}
        
    def _find_source_in_db(self, vdb: dict) -> Optional[str]:
        """
        Find source in database, handling various name formats.
        
        Args:
            vdb: Database dictionary
            
        Returns:
            Actual source name in database, or None if not found
        """
        # Direct match
        if self.source_name_input in vdb:
            return self.source_name_input
        
        # Try with 4FGL prefix
        if not self.source_name_input.startswith('4FGL'):
            test_name = f'4FGL {self.source_name_input}'
            if test_name in vdb:
                return test_name
        
        # Try without 4FGL prefix (if it has one)
        if self.source_name_input.startswith('4FGL '):
            test_name = self.source_name_input[5:]  # Remove "4FGL "
            if test_name in vdb:
                return test_name
        
        # Try case-insensitive search
        source_lower = self.source_name_input.lower()
        for db_key in vdb.keys():
            if db_key.lower() == source_lower:
                return db_key
            if db_key.lower().endswith(source_lower):
                return db_key
            if source_lower in db_key.lower():
                return db_key
        
        return None
        
    def load_source_data(self) -> bool:
        """
        Load source data from pickle file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.info_file.exists():
                print(f"Error: {self.info_file} not found")
                return False
            
            with open(self.info_file, 'rb') as f:
                vdb = pickle.load(f)
            
            # Try to find the source
            actual_name = self._find_source_in_db(vdb)
            
            if actual_name is None:
                print(f"Error: Could not find '{self.source_name_input}' in database")
                print(f"\nTried variants:")
                print(f"  • {self.source_name_input}")
                print(f"  • 4FGL {self.source_name_input}")
                if self.source_name_input.startswith('4FGL'):
                    print(f"  • {self.source_name_input[5:]}")
                
                print(f"\nSample available sources:")
                sample_sources = list(vdb.keys())[:15]
                for src in sample_sources:
                    print(f"  • {src}")
                
                return False
            
            self.source_name = actual_name
            source_dict = vdb[actual_name]
            
            # Convert light curve to DataFrame if needed
            lc = source_dict['light_curve']
            if isinstance(lc, dict):
                self.light_curve_df = pd.DataFrame.from_dict(lc, orient='index')
            else:
                self.light_curve_df = lc
            
            self.source_data = {
                'name': self.source_name,
                'light_curve': self.light_curve_df,
                'nearby': source_dict.get('nearby', None),
                'fft_peaks': source_dict.get('fft_peaks', None),
                'nbb': len(self.light_curve_df) if self.light_curve_df is not None else 0,
            }
            
            print(f"✓ Successfully loaded: {self.source_name}")
            print(f"  (Input was: {self.source_name_input})")
            print(f"  Number of Bayesian Blocks: {self.source_data['nbb']}")
            
            return True
            
        except Exception as e:
            print(f"Error loading source data: {e}")
            return False
    
    def analyze_light_curve_structure(self) -> Dict:
        """
        Analyze the structure of the light curve (number of blocks, etc).
        
        Returns:
            Dictionary with structural analysis
        """
        if self.light_curve_df is None:
            return {}
        
        lc = self.light_curve_df
        num_blocks = len(lc)
        
        # Extract flux values
        flux_values = lc.flux.values
        flux_array = np.array([
            fv[0] if hasattr(fv, '__len__') and not isinstance(fv, (str, bytes)) else fv
            for fv in flux_values
        ], dtype=float)
        
        # Extract errors
        errors = lc.errors.values
        error_lower = np.array([e[0] if isinstance(e, (list, tuple)) else e for e in errors], dtype=float)
        error_upper = np.array([e[1] if isinstance(e, (list, tuple)) else e for e in errors], dtype=float)
        
        # Time information
        times = lc.t.values
        time_widths = lc.tw.values
        
        analysis = {
            'num_blocks': num_blocks,
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
        
        self.analysis_results['structure'] = analysis
        return analysis
    
    def classify_block_structure(self) -> str:
        """
        Classify source as single-step (2 blocks) or multi-step (3+ blocks).
        
        Returns:
            Classification string
        """
        num_blocks = self.source_data['nbb']
        
        if num_blocks == 1:
            return "Single block (no variability)"
        elif num_blocks == 2:
            return "Single-step (one transition)"
        elif num_blocks == 3:
            return "Two-step (two transitions)"
        else:
            return f"Multi-step ({num_blocks-1} transitions)"
    
    def calculate_flux_ratios(self, margin_weeks: int = 50) -> Dict:
        """
        Calculate flux ratios between blocks, following agnsteps methodology.
        
        Args:
            margin_weeks: Minimum block width (in weeks) to consider valid
            
        Returns:
            Dictionary with ratio analysis
        """
        lc = self.light_curve_df
        if lc is None or len(lc) < 2:
            return {}
        
        flux_values = np.array([
            fv[0] if hasattr(fv, '__len__') and not isinstance(fv, (str, bytes)) else fv
            for fv in lc.flux.values
        ], dtype=float)
        
        time_widths = lc.tw.values / 7  # Convert to weeks
        
        ratios = {}
        
        if len(flux_values) == 2:
            # Single step: calculate before/after ratio
            a, b = flux_values[0], flux_values[1]
            w1, w2 = time_widths[0], time_widths[1]
            
            ratios['type'] = 'single_step'
            ratios['flux_before'] = a
            ratios['flux_after'] = b
            ratios['flux_ratio'] = b / a if a != 0 else np.inf
            ratios['log_ratio'] = np.log10(ratios['flux_ratio'])
            ratios['width_before_weeks'] = w1
            ratios['width_after_weeks'] = w2
            ratios['meets_margin'] = (w1 >= margin_weeks) and (w2 >= margin_weeks)
            ratios['transition_type'] = 'up' if b > a else 'down'
            ratios['transition_magnitude'] = abs(b - a) / a if a != 0 else np.inf
            
        elif len(flux_values) == 3:
            # Two steps: calculate overall and bump
            a, b, c = flux_values[0], flux_values[1], flux_values[2]
            w1, w2, w3 = time_widths[0], time_widths[1], time_widths[2]
            
            ratios['type'] = 'two_steps'
            ratios['flux_first'] = a
            ratios['flux_bump'] = b
            ratios['flux_last'] = c
            ratios['overall_ratio'] = c / a if a != 0 else np.inf
            ratios['bump_ratio'] = 2 * b / (a + c) if (a + c) != 0 else np.inf
            ratios['log_overall_ratio'] = np.log10(ratios['overall_ratio'])
            ratios['log_bump_ratio'] = np.log10(ratios['bump_ratio'])
            ratios['width_first_weeks'] = w1
            ratios['width_bump_weeks'] = w2
            ratios['width_last_weeks'] = w3
            ratios['meets_margin'] = (w1 >= margin_weeks) and (w3 >= margin_weeks)
            ratios['bump_type'] = 'flare' if b > max(a, c) else 'dip'
            
        else:
            # Multi-step: analyze all transitions
            ratios['type'] = 'multi_step'
            ratios['num_blocks'] = len(flux_values)
            
            # Calculate all consecutive ratios
            consecutive_ratios = []
            for i in range(len(flux_values) - 1):
                if flux_values[i] != 0:
                    consecutive_ratios.append(flux_values[i+1] / flux_values[i])
            
            ratios['consecutive_ratios'] = consecutive_ratios
            ratios['overall_ratio'] = flux_values[-1] / flux_values[0] if flux_values[0] != 0 else np.inf
            ratios['log_ratios'] = [np.log10(r) for r in consecutive_ratios]
        
        self.analysis_results['ratios'] = ratios
        return ratios
    
    def print_detailed_analysis(self, margin_weeks: int = 50) -> None:
        """
        Print comprehensive analysis of the source.
        
        Args:
            margin_weeks: Minimum block width for validity
        """
        if self.light_curve_df is None:
            print("Error: Light curve not loaded")
            return
        
        # Analyze structure
        structure = self.analyze_light_curve_structure()
        ratios = self.calculate_flux_ratios(margin_weeks)
        classification = self.classify_block_structure()
        
        print("\n" + "="*80)
        print(f"DETAILED ANALYSIS: {self.source_name}")
        print("="*80 + "\n")
        
        # Basic information
        print(f"Classification: {classification}\n")
        
        # Structure analysis
        print(f"Structure Analysis:")
        print(f"  Number of Bayesian Blocks: {structure['num_blocks']}")
        print(f"  Mean Flux: {structure['mean_flux']:.4f}")
        print(f"  Std Dev Flux: {structure['std_flux']:.4f}")
        print(f"  Flux Range: [{structure['min_flux']:.4f}, {structure['max_flux']:.4f}]")
        print(f"  Variability: {structure['max_flux'] / structure['min_flux']:.2f}x\n")
        
        # Time span
        times = structure['times']
        total_span = times[-1] - times[0] + structure['time_widths'][-1]
        print(f"Time Coverage:")
        print(f"  Start (MJD): {times[0]:.1f}")
        print(f"  End (MJD): {times[-1]:.1f}")
        print(f"  Total Span: {total_span:.1f} days ({total_span/365.25:.1f} years)\n")
        
        # Block-by-block information
        print(f"Block-by-Block Information:")
        print(f"{'Block':>6} {'Time (MJD)':>15} {'Width (days)':>15} {'Flux':>12} {'Error':>12}")
        print("─" * 70)
        
        for i, (t, w, f, el, eu) in enumerate(zip(
            structure['times'],
            structure['time_widths'],
            structure['flux_values'],
            structure['error_lower'],
            structure['error_upper']
        )):
            print(f"{i+1:>6} {t:>15.1f} {w:>15.1f} {f:>12.4f} +{eu:.4f}/-{el:.4f}")
        
        print("\n" + "─"*80 + "\n")
        
        # Ratio analysis
        print(f"Flux Ratio Analysis (margin={margin_weeks} weeks):\n")
        
        if ratios['type'] == 'single_step':
            print(f"  Type: Single-Step Source")
            print(f"  Before → After: {ratios['flux_before']:.4f} → {ratios['flux_after']:.4f}")
            print(f"  Flux Ratio: {ratios['flux_ratio']:.4f} ({ratios['flux_ratio']:.2f}x)")
            print(f"  Log Ratio: {ratios['log_ratio']:.4f}")
            print(f"  Transition: {ratios['transition_type'].upper()} "
                  f"(magnitude: {ratios['transition_magnitude']:.2f}x)")
            print(f"  Block 1 Width: {ratios['width_before_weeks']:.1f} weeks")
            print(f"  Block 2 Width: {ratios['width_after_weeks']:.1f} weeks")
            print(f"  Margin Check: {'✓ PASS' if ratios['meets_margin'] else '✗ FAIL'}")
            
        elif ratios['type'] == 'two_steps':
            print(f"  Type: Two-Step Source (Flare/Dip Pattern)")
            print(f"  Flux Profile: {ratios['flux_first']:.4f} → {ratios['flux_bump']:.4f} → {ratios['flux_last']:.4f}")
            print(f"  Overall Ratio (first→last): {ratios['overall_ratio']:.4f} "
                  f"({ratios['log_overall_ratio']:+.4f} in log space)")
            print(f"  Bump Ratio (prominence): {ratios['bump_ratio']:.4f} "
                  f"({ratios['log_bump_ratio']:+.4f} in log space)")
            print(f"  Bump Type: {ratios['bump_type'].upper()}")
            print(f"  Block Widths: {ratios['width_first_weeks']:.1f}w → {ratios['width_bump_weeks']:.1f}w → {ratios['width_last_weeks']:.1f}w")
            print(f"  Margin Check: {'✓ PASS' if ratios['meets_margin'] else '✗ FAIL'}")
            
        else:
            print(f"  Type: Multi-Step Source ({ratios['num_blocks']} blocks)")
            print(f"  Overall Ratio: {ratios['overall_ratio']:.4f}")
            print(f"  Consecutive Ratios: {[f'{r:.3f}' for r in ratios['consecutive_ratios']]}")
            print(f"  Log Ratios: {[f'{r:+.3f}' for r in ratios['log_ratios']]}")
        
        print("\n" + "="*80 + "\n")
    
    def plot_light_curve(self, figsize: Tuple[float, float] = (12, 6)) -> plt.Figure:
        """
        Create detailed light curve plot.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        if self.light_curve_df is None:
            print("Error: Light curve not loaded")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        structure = self.analysis_results.get('structure', self.analyze_light_curve_structure())
        
        times = structure['times']
        fluxes = structure['flux_values']
        errors_lower = structure['error_lower']
        errors_upper = structure['error_upper']
        widths = structure['time_widths']
        
        # Plot with error bars
        ax.errorbar(times, fluxes, 
                   yerr=[errors_lower, errors_upper],
                   xerr=widths/2,
                   fmt='o-', markersize=8, linewidth=2, 
                   capsize=5, capthick=2,
                   color='steelblue', ecolor='steelblue', alpha=0.7,
                   label='Bayesian Block measurements')
        
        # Add step function
        x_steps = [times[0] - widths[0]/2]
        y_steps = [fluxes[0]]
        
        for i in range(len(times)):
            x_steps.append(times[i] + widths[i]/2)
            y_steps.append(fluxes[i])
            if i < len(times) - 1:
                x_steps.append(times[i+1] - widths[i+1]/2)
                y_steps.append(fluxes[i+1])
        
        ax.step(x_steps, y_steps, where='mid', linewidth=2.5, 
               color='coral', alpha=0.8, label='Block profile')
        
        # Styling
        ax.set_xlabel('Time (MJD)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Flux (relative)', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.source_name}\nLight Curve Analysis', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add vertical lines at transitions
        for i in range(len(times) - 1):
            transition_time = times[i] + widths[i]/2
            ax.axvline(transition_time, color='red', linestyle='--', 
                      alpha=0.4, linewidth=1)
        
        plt.tight_layout()
        return fig
    
    def plot_ratio_analysis(self, figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
        """
        Create flux ratio visualization.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        if self.light_curve_df is None:
            print("Error: Light curve not loaded")
            return None
        
        ratios = self.analysis_results.get('ratios', self.calculate_flux_ratios())
        structure = self.analysis_results.get('structure', self.analyze_light_curve_structure())
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Flux values
        ax1 = axes[0]
        blocks = np.arange(1, structure['num_blocks'] + 1)
        colors = ['red' if f == structure['min_flux'] else 
                 'green' if f == structure['max_flux'] else 'steelblue'
                 for f in structure['flux_values']]
        
        ax1.bar(blocks, structure['flux_values'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_xlabel('Bayesian Block', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Flux (relative)', fontsize=11, fontweight='bold')
        ax1.set_title('Flux per Block', fontsize=12, fontweight='bold')
        ax1.set_xticks(blocks)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Ratios
        ax2 = axes[1]
        
        if ratios['type'] == 'single_step':
            ratio_data = [ratios['flux_ratio']]
            ratio_labels = ['After/Before']
            colors_ratio = ['green' if ratios['flux_ratio'] > 1 else 'red']
            
            ax2.bar(ratio_labels, ratio_data, color=colors_ratio, alpha=0.7, 
                   edgecolor='black', linewidth=2, width=0.4)
            ax2.axhline(1, color='gray', linestyle='--', linewidth=2, alpha=0.5)
            ax2.set_ylabel('Flux Ratio', fontsize=11, fontweight='bold')
            ax2.set_title('Flux Transition Ratio', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, max(1.5, max(ratio_data) * 1.1))
            
        elif ratios['type'] == 'two_steps':
            ratio_labels = ['Overall\n(First→Last)', 'Bump\nProminence']
            ratio_data = [ratios['overall_ratio'], ratios['bump_ratio']]
            colors_ratio = ['steelblue', 'orange']
            
            bars = ax2.bar(ratio_labels, ratio_data, color=colors_ratio, alpha=0.7,
                          edgecolor='black', linewidth=2, width=0.6)
            ax2.axhline(1, color='gray', linestyle='--', linewidth=2, alpha=0.5)
            ax2.set_ylabel('Flux Ratio', fontsize=11, fontweight='bold')
            ax2.set_title('Flux Ratios (Multi-Block)', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, max(ratio_data) * 1.2)
            
            # Add value labels on bars
            for bar, value in zip(bars, ratio_data):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        else:
            n_transitions = len(ratios['consecutive_ratios'])
            trans_labels = [f'T{i+1}' for i in range(n_transitions)]
            
            colors_ratio = ['green' if r > 1 else 'red' for r in ratios['consecutive_ratios']]
            ax2.bar(trans_labels, ratios['consecutive_ratios'], color=colors_ratio, 
                   alpha=0.7, edgecolor='black', linewidth=2)
            ax2.axhline(1, color='gray', linestyle='--', linewidth=2, alpha=0.5)
            ax2.set_ylabel('Flux Ratio', fontsize=11, fontweight='bold')
            ax2.set_title('All Consecutive Transitions', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self, margin_weeks: int = 50) -> str:
        """
        Generate a text summary report.
        
        Args:
            margin_weeks: Minimum block width for validity
            
        Returns:
            Formatted summary string
        """
        if self.light_curve_df is None:
            return "Error: Source data not loaded"
        
        structure = self.analysis_results.get('structure', self.analyze_light_curve_structure())
        ratios = self.analysis_results.get('ratios', self.calculate_flux_ratios(margin_weeks))
        classification = self.classify_block_structure()
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    AGN STEPS ANALYSIS REPORT                               ║
║                          {self.source_name:<52}║
╚════════════════════════════════════════════════════════════════════════════╝

CLASSIFICATION
──────────────
{classification}

LIGHT CURVE STRUCTURE
─────────────────────
  • Number of Blocks: {structure['num_blocks']}
  • Flux Range: {structure['min_flux']:.4f} - {structure['max_flux']:.4f}
  • Mean Flux: {structure['mean_flux']:.4f} ± {structure['std_flux']:.4f}
  • Total Variability: {structure['max_flux'] / structure['min_flux']:.2f}x

FLUX RATIOS
───────────
"""
        
        if ratios['type'] == 'single_step':
            report += f"""
  • Transition Type: {ratios['transition_type'].upper()}
  • Flux Before: {ratios['flux_before']:.4f}
  • Flux After: {ratios['flux_after']:.4f}
  • Ratio (After/Before): {ratios['flux_ratio']:.4f} ({ratios['flux_ratio']:.2f}x)
  • Log Ratio: {ratios['log_ratio']:+.4f}
  • Block 1 Duration: {ratios['width_before_weeks']:.1f} weeks
  • Block 2 Duration: {ratios['width_after_weeks']:.1f} weeks
  • Margin ({margin_weeks}w) Check: {'✓ PASS' if ratios['meets_margin'] else '✗ FAIL'}
"""
        
        elif ratios['type'] == 'two_steps':
            report += f"""
  • Pattern: {ratios['bump_type'].upper()} in middle block
  • First Block Flux: {ratios['flux_first']:.4f}
  • Middle Block Flux: {ratios['flux_bump']:.4f}
  • Last Block Flux: {ratios['flux_last']:.4f}
  • Overall Ratio (First→Last): {ratios['overall_ratio']:.4f}
  • Bump Prominence: {ratios['bump_ratio']:.4f}
  • Log Overall: {ratios['log_overall_ratio']:+.4f}
  • Log Bump: {ratios['log_bump_ratio']:+.4f}
  • Block Durations: {ratios['width_first_weeks']:.1f}w → {ratios['width_bump_weeks']:.1f}w → {ratios['width_last_weeks']:.1f}w
  • Margin ({margin_weeks}w) Check: {'✓ PASS' if ratios['meets_margin'] else '✗ FAIL'}
"""
        
        report += f"""
ASTROPHYSICAL INTERPRETATION
────────────────────────────
"""
        
        if structure['num_blocks'] == 2:
            if ratios['flux_ratio'] > 1:
                report += f"""
  The source exhibited a sustained flux increase by a factor of {ratios['flux_ratio']:.2f}
  between two stable states. This is consistent with a change in accretion state
  or jet configuration lasting years.
"""
            else:
                report += f"""
  The source exhibited a sustained flux decrease by a factor of {1/ratios['flux_ratio']:.2f}
  between two stable states. This is consistent with a dimming event or reduction
  in accretion activity lasting years.
"""
        
        elif structure['num_blocks'] == 3:
            report += f"""
  The source shows evidence of a transient {ratios['bump_type']} superimposed on
  long-term stability. The middle block ({ratios['width_bump_weeks']:.1f} weeks) represents
  a temporary deviation from the baseline, with {"increased" if ratios["bump_type"] == "flare" else "decreased"}
  activity by a factor of {ratios['bump_ratio']:.2f}.
"""
        
        report += """
════════════════════════════════════════════════════════════════════════════
"""
        
        return report


# Example usage and main execution
if __name__ == "__main__":
    
    # Create analyzer for source - can use any of these formats:
    # 'J0725.8-0054'
    # 'J0725.8-0054'
    # '4FGL J0725.8-0054'
    analyzer = AGNStepAnalyzer(
        source_name='J0725.8-0054',  # No 4FGL prefix needed!
        info_file='files/source_info_v1.pkl'
    )
    
    # Load source data
    if analyzer.load_source_data():
        
        # Print detailed analysis
        analyzer.print_detailed_analysis(margin_weeks=50)
        
        # Generate and print summary report
        report = analyzer.generate_summary_report(margin_weeks=50)
        print(report)
        
        # Create visualizations
        print("Generating visualizations...")
        
        # Light curve plot
        fig1 = analyzer.plot_light_curve()
        source_safe_name = analyzer.source_name.replace(' ', '_').replace('/', '_')
        plt.savefig(f'{source_safe_name}_light_curve.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved light curve plot")
        
        # Ratio analysis plot
        fig2 = analyzer.plot_ratio_analysis()
        plt.savefig(f'{source_safe_name}_ratios.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved ratio analysis plot")
        
        plt.show()
        
    else:
        print("Failed to load source data. Check file path and source name.")