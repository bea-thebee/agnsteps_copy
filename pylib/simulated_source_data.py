"""
Generate synthetic source variability data similar to source_info_v3.pkl format.

This module creates realistic simulated Fermi gamma-ray source data with:
- Light curves with Bayesian block partitions
- Realistic flux distributions and variability metrics
- Source catalog properties (coordinates, spectral parameters, etc.)
- Multiple realizations for Monte Carlo studies

The data structure mirrors VarDB format from data_setup.py, allowing
easy integration with existing analysis pipelines.

Key Features:
1. Realistic Light Curves: Generates Bayesian block structures with variable flux levels
2. Catalog Properties: Realistic distributions for TS, flux, spectral indices, etc.
3. Multiple Realizations: Generate dozens of independent MC samples easily
4. Reproducible: Full seed control for deterministic generation
5. Compatible Format: Matches source_info_v3.pkl structure
6. Configurable: Custom parameters for light curves and flux variations
7. Statistical Output: Built-in DataFrame conversion and summary statistics
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import OrderedDict


@dataclass
class FluxParameters:
    """Parameters for generating realistic flux values"""
    mean_flux: float = 1.0  # Relative to reference
    flux_std: float = 0.3   # Standard deviation
    min_flux: float = 0.1   # Minimum flux
    max_flux: float = 10.0  # Maximum flux


@dataclass
class LightCurveParameters:
    """Parameters for light curve generation"""
    n_bins: int = 100  # Number of time bins
    mean_bin_width: float = 30  # Days
    std_bin_width: float = 10   # Days
    prob_new_block: float = 0.05  # Probability of Bayesian block boundary
    variability_scale: float = 0.5  # Scaling factor for variability


class SimulatedLightCurve:
    """Generate a realistic simulated light curve"""
    
    def __init__(self, params: LightCurveParameters = None, seed: int = None):
        """
        Initialize light curve generator
        
        Parameters
        ----------
        params : LightCurveParameters
            Configuration parameters
        seed : int, optional
            Random seed for reproducibility
        """
        self.params = params or LightCurveParameters()
        if seed is not None:
            np.random.seed(seed)
    
    def generate_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate bin times and widths following realistic patterns
        
        Returns
        -------
        t : np.ndarray
            Time bin centers
        tw : np.ndarray
            Time bin widths
        """
        widths = np.random.normal(
            self.params.mean_bin_width,
            self.params.std_bin_width,
            self.params.n_bins
        )
        widths = np.clip(widths, 1, 100)  # Realistic bounds
        
        # Accumulate to get bin centers
        cumsum = np.cumsum(widths)
        t = cumsum - widths / 2
        
        return t, widths
    
    def generate_blocks(self, t: np.ndarray) -> np.ndarray:
        """
        Determine Bayesian block boundaries using a stochastic process
        
        Parameters
        ----------
        t : np.ndarray
            Time bin centers
        
        Returns
        -------
        block_indices : np.ndarray
            Indices where new blocks start
        """
        # Start with first bin
        blocks = [0]
        
        # Stochastically add block boundaries
        for i in range(1, len(t)):
            if np.random.random() < self.params.prob_new_block:
                blocks.append(i)
        
        # Always end with last bin
        if blocks[-1] != len(t) - 1:
            blocks.append(len(t) - 1)
        
        return np.array(blocks)
    
    def generate_flux_values(self, block_indices: np.ndarray, 
                           flux_params: FluxParameters = None) -> np.ndarray:
        """
        Generate flux values with realistic variability structure
        
        Parameters
        ----------
        block_indices : np.ndarray
            Indices of block boundaries
        flux_params : FluxParameters
            Flux parameters
        
        Returns
        -------
        flux : np.ndarray
            Flux values for each bin
        """
        flux_params = flux_params or FluxParameters()
        n_blocks = len(block_indices)
        
        # Generate block-level flux variations
        block_fluxes = np.random.lognormal(
            np.log(flux_params.mean_flux),
            flux_params.flux_std,
            n_blocks
        )
        block_fluxes = np.clip(
            block_fluxes,
            flux_params.min_flux,
            flux_params.max_flux
        )
        
        # Assign block fluxes to individual bins
        flux = np.zeros(len(block_indices))
        for i, idx in enumerate(block_indices):
            flux[i] = block_fluxes[i]
        
        # Add bin-level noise
        flux *= np.random.normal(1.0, 0.1, len(flux))
        flux = np.clip(flux, flux_params.min_flux, flux_params.max_flux)
        
        return flux
    
    def generate_errors(self, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic flux errors (lower and upper bounds)
        
        Parameters
        ----------
        flux : np.ndarray
            Flux values
        
        Returns
        -------
        error_lower : np.ndarray
            Lower error bounds
        error_upper : np.ndarray
            Upper error bounds
        """
        # Errors scale with flux and have some baseline
        relative_error = np.random.uniform(0.05, 0.2, len(flux))
        error_lower = flux * relative_error
        error_upper = flux * relative_error * np.random.uniform(0.8, 1.2, len(flux))
        
        return error_lower, error_upper
    
    def generate(self, flux_params: FluxParameters = None) -> Dict:
        """
        Generate a complete light curve
        
        Returns
        -------
        lc_dict : Dict
            Light curve data with keys: 't', 'tw', 'flux', 'errors'
        """
        flux_params = flux_params or FluxParameters()
        
        # Generate time structure
        t, tw = self.generate_bins()
        
        # Generate blocks
        block_indices = self.generate_blocks(t)
        
        # Generate fluxes
        flux = self.generate_flux_values(block_indices, flux_params)
        
        # Generate errors
        error_lower, error_upper = self.generate_errors(flux)
        errors = list(zip(error_lower, error_upper))
        
        return {
            't': t.tolist(),
            'tw': tw.tolist(),
            'flux': flux.tolist(),
            'errors': errors
        }


class SourcePropertyGenerator:
    """Generate realistic source catalog properties"""
    
    ASSOCIATIONS = ['bll', 'fsrq', 'bcu', 'psr', 'unid', 'other']
    
    @staticmethod
    def generate_coordinates(n_sources: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random galactic coordinates
        
        Returns
        -------
        glon : np.ndarray
            Galactic longitude
        glat : np.ndarray
            Galactic latitude
        """
        if seed is not None:
            np.random.seed(seed)
        
        glon = np.random.uniform(0, 360, n_sources)
        glat = np.random.uniform(-90, 90, n_sources)
        
        return glon, glat
    
    @staticmethod
    def generate_test_statistic(n_sources: int, seed: int = None) -> np.ndarray:
        """
        Generate realistic TS values (typically log-normal distributed)
        
        Returns
        -------
        ts : np.ndarray
            Test statistic values
        """
        if seed is not None:
            np.random.seed(seed)
        
        # TS values typically range from ~25 to 1000+
        ts = np.random.lognormal(3.5, 1.5, n_sources)
        return np.clip(ts, 25, 2000)
    
    @staticmethod
    def generate_eflux100(n_sources: int, seed: int = None) -> np.ndarray:
        """
        Generate 100 MeV integrated flux (E>100 MeV)
        
        Returns
        -------
        eflux100 : np.ndarray
            Flux in erg cm^-2 s^-1
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Typical range 1e-13 to 1e-11 erg/cm2/s
        log_eflux = np.random.uniform(-13, -11, n_sources)
        return 10.0 ** log_eflux
    
    @staticmethod
    def generate_spectral_index(n_sources: int, seed: int = None) -> np.ndarray:
        """
        Generate photon spectral indices (typically 1.5 - 2.5)
        
        Returns
        -------
        pindex : np.ndarray
            Photon spectral index
        """
        if seed is not None:
            np.random.seed(seed)
        
        return np.random.normal(2.0, 0.4, n_sources).clip(1.0, 3.0)
    
    @staticmethod
    def generate_variability_index(n_sources: int, seed: int = None) -> np.ndarray:
        """
        Generate BBvar index (Bayesian Block variability metric)
        
        Returns
        -------
        bbvar : np.ndarray
            BBvar values
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Most sources are non-variable (bbvar ~1), some are variable (bbvar > 10)
        is_variable = np.random.random(n_sources) < 0.1
        bbvar = np.where(
            is_variable,
            np.random.lognormal(2, 1.5, n_sources),  # Variable sources
            np.random.uniform(0.5, 3, n_sources)      # Non-variable
        )
        return np.clip(bbvar, 0.1, 200)
    
    @staticmethod
    def generate_associations(n_sources: int, seed: int = None) -> List[str]:
        """
        Generate source type associations with realistic proportions
        
        Returns
        -------
        associations : List[str]
            Source type for each source
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Realistic proportions (based on 4FGL-DR4)
        weights = [0.35, 0.20, 0.10, 0.05, 0.20, 0.10]  # bll, fsrq, bcu, psr, unid, other
        associations = np.random.choice(
            SourcePropertyGenerator.ASSOCIATIONS,
            size=n_sources,
            p=weights
        )
        return associations.tolist()


class SimulatedSourceDatabase:
    """Generate a complete simulated source database"""
    
    def __init__(self, n_sources: int = 100, seed: int = None):
        """
        Initialize the database generator
        
        Parameters
        ----------
        n_sources : int
            Number of sources to generate
        seed : int, optional
            Random seed for reproducibility
        """
        self.n_sources = n_sources
        self.seed = seed
        self.sources = OrderedDict()
        self.metadata = {}
    
    def generate(self, 
                 lc_params: LightCurveParameters = None,
                 flux_params: FluxParameters = None,
                 verbose: bool = True) -> 'SimulatedSourceDatabase':
        """
        Generate a complete simulated database
        
        Parameters
        ----------
        lc_params : LightCurveParameters
            Light curve generation parameters
        flux_params : FluxParameters
            Flux generation parameters
        verbose : bool
            Print progress information
        
        Returns
        -------
        self : SimulatedSourceDatabase
        """
        lc_params = lc_params or LightCurveParameters()
        flux_params = flux_params or FluxParameters()
        
        if verbose:
            print(f"Generating {self.n_sources} simulated sources...")
        
        # Generate source catalog properties
        glon, glat = SourcePropertyGenerator.generate_coordinates(
            self.n_sources, seed=self.seed
        )
        ts = SourcePropertyGenerator.generate_test_statistic(
            self.n_sources, seed=self.seed
        )
        eflux100 = SourcePropertyGenerator.generate_eflux100(
            self.n_sources, seed=self.seed
        )
        pindex = SourcePropertyGenerator.generate_spectral_index(
            self.n_sources, seed=self.seed
        )
        bbvar = SourcePropertyGenerator.generate_variability_index(
            self.n_sources, seed=self.seed
        )
        associations = SourcePropertyGenerator.generate_associations(
            self.n_sources, seed=self.seed
        )
        
        # Generate light curves
        for i in range(self.n_sources):
            source_id = f'4FGL J{i:04d}'
            
            # Generate light curve with per-source seed
            lc_gen = SimulatedLightCurve(lc_params, seed=self.seed + i if self.seed else None)
            light_curve = lc_gen.generate(flux_params)
            
            # Nearby sources (simplified)
            nearby = [{'jname': f'nearby_{j}', 'sep': np.random.uniform(0.1, 5)}
                     for j in range(np.random.randint(0, 5))]
            
            # Poisson fit results (simplified as tuples)
            poisson_fits = [
                (1.0 + np.random.normal(0, 0.1), np.random.uniform(0.05, 0.15))
                for _ in range(len(light_curve['flux']))
            ]
            
            self.sources[source_id] = {
                'light_curve': light_curve,
                'nearby': nearby if nearby else None,
                'poisson_fits': poisson_fits,
                'ts': float(ts[i]),
                'eflux100': float(eflux100[i]),
                'pindex': float(pindex[i]),
                'bbvar': int(bbvar[i]),
                'variability': int(np.random.randint(1, 100)),
                'association': associations[i],
                'glon': float(glon[i]),
                'glat': float(glat[i]),
            }
            
            if verbose and (i + 1) % max(1, self.n_sources // 10) == 0:
                print(f"  Generated {i + 1}/{self.n_sources} sources")
        
        # Store metadata
        self.metadata = {
            'n_sources': self.n_sources,
            'seed': self.seed,
            'lc_params': asdict(lc_params),
            'flux_params': asdict(flux_params),
        }
        
        return self
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format matching VarDB"""
        return dict(self.sources)
    
    def save(self, filepath: str) -> None:
        """
        Save simulated database to pickle file
        
        Parameters
        ----------
        filepath : str
            Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.to_dict(), f)
        
        print(f"Saved simulated database to {filepath}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to DataFrame for analysis
        
        Returns
        -------
        df : pd.DataFrame
            Source properties
        """
        rows = []
        for source_id, data in self.sources.items():
            rows.append({
                'source_id': source_id,
                'ts': data['ts'],
                'eflux100': data['eflux100'],
                'pindex': data['pindex'],
                'bbvar': data['bbvar'],
                'variability': data['variability'],
                'association': data['association'],
                'glon': data['glon'],
                'glat': data['glat'],
                'nbb': len(data['light_curve']['flux']) if data['light_curve'] else 0,
                'near': len(data['nearby']) if data['nearby'] else 0,
            })
        
        return pd.DataFrame(rows)


def generate_multiple_realizations(n_realizations: int, 
                                   n_sources: int = 100,
                                   output_dir: str = 'simulated_data',
                                   seed_base: int = 0) -> List[str]:
    """
    Generate multiple independent realizations of the simulated database.
    Useful for Monte Carlo studies and uncertainty quantification.
    
    Parameters
    ----------
    n_realizations : int
        Number of independent realizations to generate
    n_sources : int
        Number of sources per realization
    output_dir : str
        Directory to save files
    seed_base : int
        Base random seed (incremented for each realization)
    
    Returns
    -------
    filepaths : List[str]
        Paths to saved realization files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepaths = []
    
    for i in range(n_realizations):
        print(f"\n{'='*60}")
        print(f"Generating realization {i+1}/{n_realizations}")
        print(f"{'='*60}")
        
        # Create new database with different seed
        seed = seed_base + i * 1000
        db = SimulatedSourceDatabase(n_sources=n_sources, seed=seed)
        
        # Generate with potentially different parameters
        db.generate(verbose=True)
        
        # Save
        filepath = output_path / f'simulated_sources_v3_realization_{i:02d}.pkl'
        db.save(str(filepath))
        filepaths.append(str(filepath))
        
        # Summary statistics
        df = db.to_dataframe()
        print(f"\nRealization {i+1} Summary:")
        print(f"  Total sources: {len(df)}")
        print(f"  Mean TS: {df.ts.mean():.1f}")
        print(f"  Mean bbvar: {df.bbvar.mean():.1f}")
        print(f"  Variable sources (bbvar>10): {sum(df.bbvar > 10)}")
        print(f"  Association distribution:")
        for assoc, count in df.association.value_counts().items():
            print(f"    {assoc}: {count}")
    
    return filepaths


# Example usage and demonstration
if __name__ == '__main__':
    print("Simulated Source Data Generation")
    print("="*60)
    
    # Example 1: Generate a single database
    print("\nExample 1: Single database with 50 sources")
    db_single = SimulatedSourceDatabase(n_sources=50, seed=42)
    db_single.generate()
    
    # Convert to DataFrame for inspection
    df_single = db_single.to_dataframe()
    print("\nDatabase Summary:")
    print(df_single.describe())
    
    # Example 2: Generate multiple realizations
    print("\n" + "="*60)
    print("Example 2: Multiple realizations for Monte Carlo")
    filepaths = generate_multiple_realizations(
        n_realizations=3,
        n_sources=100,
        output_dir='./simulated_sources',
        seed_base=1000
    )
    
    print("\n" + "="*60)
    print("Generated files:")
    for fp in filepaths:
        print(f"  {fp}")
    
    # Example 3: Custom parameters
    print("\n" + "="*60)
    print("Example 3: Custom light curve parameters")
    custom_lc = LightCurveParameters(
        n_bins=50,  # Fewer bins
        mean_bin_width=50,
        prob_new_block=0.10,  # More blocks expected
    )
    custom_flux = FluxParameters(
        mean_flux=1.5,
        flux_std=0.6,  # Higher variability
    )
    
    db_custom = SimulatedSourceDatabase(n_sources=25, seed=999)
    db_custom.generate(lc_params=custom_lc, flux_params=custom_flux)
    db_custom.save('./simulated_sources/custom_params.pkl')
    
    print("\nCustom parameters database saved!")