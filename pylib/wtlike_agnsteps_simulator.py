"""
WTLike-based AGN Steps Simulator
Generates synthetic Fermi-LAT-like photon data with realistic characteristics
that can be analyzed using Bayesian Blocks to detect AGN flux transitions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.stats import poisson
from astropy.stats.bayesian_blocks import FitnessFunc


@dataclass
class PhotonEvent:
    """Represents a single gamma-ray photon detection"""
    time_mjd: float      # Modified Julian Date
    energy_mev: float    # Energy in MeV
    weight: float        # Probability of being source photon (0-1)
    exposure: float      # Effective exposure at this time


class CellData:
    """
    Represents binned photon data in time cells (time intervals).
    Mimics wtlike.cell_data.CellData structure.
    """
    
    def __init__(self, photons: List[PhotonEvent], time_bins: Tuple[float, float, float]):
        """
        Initialize cell data.
        
        Args:
            photons: List of PhotonEvent objects
            time_bins: (start_mjd, end_mjd, bin_width_days)
        """
        self.photons = photons
        self.start_time, self.end_time, self.bin_width = time_bins
        self.cells = self._create_cells()
        
    def _create_cells(self) -> pd.DataFrame:
        """Create binned cells from photon data"""
        # Create time bins
        nbins = int(np.ceil((self.end_time - self.start_time) / self.bin_width))
        bin_edges = np.linspace(self.start_time, self.end_time, nbins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        
        cells_data = []
        
        for i, (center, width) in enumerate(zip(bin_centers, bin_widths)):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            
            # Get photons in this bin
            photons_in_bin = [
                p for p in self.photons 
                if bin_start <= p.time_mjd < bin_end
            ]
            
            if not photons_in_bin:
                continue
                
            # Extract weights and exposure
            weights = np.array([p.weight for p in photons_in_bin])
            exposures = np.array([p.exposure for p in photons_in_bin])
            
            cells_data.append({
                't': center,              # Cell center time
                'tw': width,              # Cell width
                'e': np.mean(exposures),  # Average exposure
                'n': len(photons_in_bin), # Number of photons
                'w': weights,             # Weight list
                'S': np.sum(weights),     # Expected source photons
                'B': len(photons_in_bin) - np.sum(weights),  # Background estimate
            })
        
        return pd.DataFrame(cells_data)


class CountFitness(FitnessFunc):
    """
    Fitness function for Bayesian Blocks using count data.
    Based on wtlike.bayesian.CountFitness
    """
    
    def __init__(self, lc_df: pd.DataFrame, p0: float = 0.05):
        """
        Args:
            lc_df: DataFrame with columns [t, tw, e, n, w, S, B]
            p0: False positive probability
        """
        self.p0 = p0
        self.df = lc_df
        self.N = len(lc_df)
        self.ncp_prior = self.p0_prior(self.N)
        
        # Time information
        t = lc_df['t'].values
        up = np.sign(t[1] - t[0]) if len(t) > 1 else 1
        dt = lc_df['tw'].values / 2 * up
        self.mjd = np.concatenate([t - dt, [t[-1] + dt[-1]]])
        
        self.setup()
        
    def setup(self):
        """Setup for fitness calculation"""
        # Photon counts
        self.nn = self.df['n'].values
        assert np.min(self.nn) > 0, 'Cell with no photons'
        
        # Exposure-based edges
        e = self.df['e'].values
        self.edges = np.concatenate([[0], np.cumsum(e)])
        self.block_length = self.edges[-1] - self.edges
        
    def __call__(self, R: int) -> np.ndarray:
        """Fitness function for Bayesian Blocks algorithm"""
        w_k = self.block_length[:R + 1] - self.block_length[R + 1]
        N_k = np.cumsum(self.nn[:R + 1][::-1])[::-1]
        
        # Scargle equation 26
        with np.errstate(divide='ignore', invalid='ignore'):
            result = N_k * (np.log(N_k) - np.log(w_k))
            result = np.nan_to_num(result, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        
        return result
    
    def fit(self) -> np.ndarray:
        """Find optimal partition using Bayesian Blocks"""
        N = self.N
        best = np.zeros(N, dtype=float)
        last = np.zeros(N, dtype=int)
        
        # Forward pass
        for R in range(N):
            fit_vec = self(R)
            A_R = fit_vec - self.ncp_prior
            A_R[1:] += best[:R]
            i_max = np.argmax(A_R)
            last[R] = i_max
            best[R] = A_R[i_max]
        
        # Backward pass to find changepoints
        change_points = np.zeros(N, dtype=int)
        i_cp = N
        ind = N
        while True:
            i_cp -= 1
            change_points[i_cp] = ind
            if ind == 0:
                break
            ind = last[ind - 1]
        
        change_points = change_points[i_cp:]
        return self.mjd[change_points]


class WTLikeAGNSimulator:
    """
    Simulates Fermi-LAT AGN observations with realistic characteristics
    and performs Bayesian Block analysis on light curves.
    """
    
    def __init__(self, mission_duration: float = 5475, source_rate: float = 2e-7, 
                 background_rate: float = 3.5e-7, effective_area: float = 2800):
        """
        Initialize simulator.
        
        Args:
            mission_duration: Total mission duration in days (default: 15 years)
            source_rate: Source photon rate in photons/cm²/s
            background_rate: Background rate in photons/cm²/s
            effective_area: Effective detection area in cm²
        """
        self.mission_duration = mission_duration
        self.source_rate = source_rate
        self.background_rate = background_rate
        self.effective_area = effective_area
        self.photons = []
        self.steps = []
        
    def add_flux_step(self, start_mjd: float, end_mjd: float, 
                     source_rate: float, step_name: str = '') -> None:
        """
        Add a constant flux step to the light curve.
        
        Args:
            start_mjd: Start time in MJD
            end_mjd: End time in MJD
            source_rate: Source rate during this period (photons/cm²/s)
            step_name: Optional name for this step
        """
        self.steps.append({
            'start': start_mjd,
            'end': end_mjd,
            'rate': source_rate,
            'name': step_name or f'Step {len(self.steps) + 1}'
        })
        self.steps.sort(key=lambda s: s['start'])
        
    def generate_photons(self, num_photons_target: int = 50000) -> List[PhotonEvent]:
        """
        Generate synthetic photon events based on defined steps.
        
        Args:
            num_photons_target: Target number of photons to generate
            
        Returns:
            List of PhotonEvent objects
        """
        self.photons = []
        
        # Calculate total exposure
        total_exposure = num_photons_target / (self.source_rate * self.effective_area)
        
        # Photons per step proportional to (rate * duration)
        step_contributions = []
        total_rate_days = 0
        
        for step in self.steps:
            duration = step['end'] - step['start']
            rate_days = step['rate'] * duration
            step_contributions.append(rate_days)
            total_rate_days += rate_days
        
        # Generate photons for each step
        for step_idx, step in enumerate(self.steps):
            duration = step['end'] - step['start']
            num_photons_step = int(num_photons_target * 
                                   step_contributions[step_idx] / total_rate_days)
            
            # Generate photon times uniformly in this step
            times = np.random.uniform(step['start'], step['end'], num_photons_step)
            
            # Generate energies (log-uniform between 100 MeV and 10 GeV)
            energies = np.random.uniform(np.log(100), np.log(10000), num_photons_step)
            energies = np.exp(energies)
            
            # Generate weights (probability of being source photon)
            # Higher in central region, lower at edges
            source_fraction = step['rate'] / (step['rate'] + self.background_rate)
            weights = np.random.binomial(1, source_fraction, num_photons_step)
            weights = weights * np.random.uniform(0.5, 1.0, num_photons_step)
            
            # Create exposure values
            exposures = np.ones(num_photons_step) * self.effective_area
            
            for t, e, w in zip(times, energies, weights):
                self.photons.append(PhotonEvent(
                    time_mjd=t,
                    energy_mev=e,
                    weight=w,
                    exposure=exposures[0]
                ))
        
        # Sort by time
        self.photons.sort(key=lambda p: p.time_mjd)
        
        return self.photons
    
    def create_light_curve(self, time_bins: Tuple[float, float, float]) -> CellData:
        """
        Create light curve from photons.
        
        Args:
            time_bins: (start_mjd, end_mjd, bin_width_days)
            
        Returns:
            CellData object with binned data
        """
        if not self.photons:
            raise ValueError("No photons generated. Call generate_photons() first.")
        
        return CellData(self.photons, time_bins)
    
    def apply_bayesian_blocks(self, cell_data: CellData, 
                             p0: float = 0.05) -> Tuple[np.ndarray, int]:
        """
        Apply Bayesian Blocks algorithm to light curve.
        
        Args:
            cell_data: CellData object
            p0: False positive probability
            
        Returns:
            Tuple of (block_edges_mjd, num_blocks)
        """
        if cell_data.cells.empty:
            raise ValueError("Cell data is empty")
        
        # Apply Bayesian Blocks
        fitness = CountFitness(cell_data.cells, p0=p0)
        edges = fitness.fit()
        
        num_blocks = len(edges) - 1
        
        if True:  # Verbose output
            print(f'Bayesian Blocks: partitioning {len(cell_data.cells)} cells')
            print(f'  Found {num_blocks} blocks from {len(cell_data.cells)} cells')
            print(f'  Penalty parameter (p0): {100*p0:.1f}%')
        
        return edges, num_blocks
    
    def plot_light_curve(self, cell_data: CellData, 
                        block_edges: Optional[np.ndarray] = None,
                        figsize: Tuple[float, float] = (14, 6)) -> plt.Figure:
        """
        Plot light curve with optional Bayesian Block overlay.
        
        Args:
            cell_data: CellData object
            block_edges: Optional array of block edge times
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot cells
        cells = cell_data.cells
        ax.errorbar(cells['t'], cells['S'], xerr=cells['tw']/2, 
                   fmt='o', markersize=4, alpha=0.7, label='Cells (Source counts)')
        
        # Overlay flux steps
        for step in self.steps:
            ax.axvspan(step['start'], step['end'], alpha=0.15, 
                      label=f"{step['name']}: {step['rate']:.2e}")
        
        # Overlay Bayesian Blocks
        if block_edges is not None:
            for i in range(len(block_edges) - 1):
                start, end = block_edges[i], block_edges[i + 1]
                mid = (start + end) / 2
                # Get average flux in this block
                block_cells = cells[(cells['t'] >= start) & (cells['t'] <= end)]
                if len(block_cells) > 0:
                    avg_flux = block_cells['S'].mean()
                    ax.plot([start, end], [avg_flux, avg_flux], 'r-', linewidth=2.5)
        
        ax.set_xlabel('Time (MJD)')
        ax.set_ylabel('Source Counts (Expected)')
        ax.set_title('WTLike AGN Steps: Light Curve with Bayesian Blocks')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def summary_statistics(self, cell_data: CellData, 
                          block_edges: Optional[np.ndarray] = None) -> str:
        """Generate summary statistics"""
        cells = cell_data.cells
        text = "=== WTLike AGN Steps Simulation Summary ===\n\n"
        text += f"Photons Generated: {len(self.photons):,}\n"
        text += f"Time Bins (Cells): {len(cells)}\n"
        text += f"Mission Duration: {self.mission_duration:.1f} days\n"
        text += f"Source Rate: {self.source_rate:.2e} photons/cm²/s\n"
        text += f"Background Rate: {self.background_rate:.2e} photons/cm²/s\n\n"
        
        text += "Flux Steps:\n"
        for step in self.steps:
            duration = step['end'] - step['start']
            text += f"  {step['name']}: {duration:.1f} days @ {step['rate']:.2e}\n"
        
        text += f"\nCell Statistics:\n"
        text += f"  Photons per cell: {cells['n'].mean():.1f} ± {cells['n'].std():.1f}\n"
        text += f"  Source counts: {cells['S'].mean():.1f} ± {cells['S'].std():.1f}\n"
        
        if block_edges is not None:
            num_blocks = len(block_edges) - 1
            text += f"\nBayesian Blocks:\n"
            text += f"  Number of blocks: {num_blocks}\n"
            text += f"  Compression: {len(cells)} cells → {num_blocks} blocks\n"
        
        return text


# Example usage
if __name__ == "__main__":
    print("WTLike AGN Steps Simulator - Example\n")
    
    # Create simulator
    sim = WTLikeAGNSimulator(
        mission_duration=5475,  # 15 years
        source_rate=2.0e-7,
        background_rate=3.5e-7,
        effective_area=2800
    )
    
    # Define flux steps (realistic AGN variability)
    sim.add_flux_step(54683, 55500, 1.0e-7, "Low State")
    sim.add_flux_step(55500, 55700, 3.5e-7, "High State")
    sim.add_flux_step(55700, 57500, 1.5e-7, "Intermediate")
    
    # Generate synthetic photons
    print("Generating synthetic photons...")
    photons = sim.generate_photons(num_photons_target=40000)
    print(f"Generated {len(photons)} photons\n")
    
    # Create light curve with 7-day bins
    print("Creating light curve with 7-day bins...")
    lc = sim.create_light_curve((54683, 57500, 7))
    print(f"Created {len(lc.cells)} cells\n")
    
    # Apply Bayesian Blocks
    print("Applying Bayesian Blocks algorithm...")
    block_edges, num_blocks = sim.apply_bayesian_blocks(lc, p0=0.05)
    print()
    
    # Print summary
    print(sim.summary_statistics(lc, block_edges))
    
    # Plot
    fig = sim.plot_light_curve(lc, block_edges)
    plt.tight_layout()
    plt.savefig('wtlike_agnsteps_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()