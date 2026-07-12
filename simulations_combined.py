"""
Combined simulated-source generator and Bayesian-block Monte Carlo simulation.

This module merges the functionality of two original modules:
 - simulated_source_data.py  (simulated catalogs and light curves)
 - bayesian_block_mc.py      (Monte Carlo of Bayesian block counts & visualization)

Design goals / API
- Single file containing:
    - FluxParameters, LightCurveParameters, BBCountSimulationConfig
    - SimulatedLightCurve, SimulatedSourceDatabase
    - BayesianBlockCounter, MonteCarloBlockSimulation
    - BlockCountVisualizer
- Simplified top-level API functions:
    - generate_simulated_db(...)
    - run_null_simulation(...)
    - run_injected_variability(...)
- RNG usage uses numpy.random.Generator for reproducibility (seed -> Generator).
- Clear docstrings and small CLI example under __main__.

Author: combined by assistant (based on provided sources)
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy import stats

# seaborn is optional for nicer plots
try:
    import seaborn as sns  # noqa: F401
except Exception:
    sns = None


# -------------------------
# Configuration dataclasses
# -------------------------
@dataclass
class FluxParameters:
    mean_flux: float = 1.0
    flux_std: float = 0.3
    min_flux: float = 0.1
    max_flux: float = 10.0


@dataclass
class LightCurveParameters:
    n_bins: int = 100
    mean_bin_width: float = 30.0
    std_bin_width: float = 10.0
    prob_new_block: float = 0.05
    variability_scale: float = 0.5  # currently unused, left for extensibility


@dataclass
class BBCountSimulationConfig:
    n_simulations: int = 1000
    n_sources: int = 100
    n_bins_per_lc: int = 100
    mean_bin_width: float = 30.0
    use_constant_flux: bool = True
    constant_flux_level: float = 1.0
    prob_block_per_bin: float = 0.03
    seed_base: int = 0


# -------------------------
# Light curve & catalog gen
# -------------------------
class SimulatedLightCurve:
    """
    Generator for a single simulated light curve.

    Produces:
      - t: bin centers (days)
      - tw: bin widths
      - flux: per-bin flux values (length n_bins)
      - errors: per-bin 1-sigma errors (length n_bins)
      - block_indices: indices where blocks start (including 0 and last index)
    """

    def __init__(self, params: Optional[LightCurveParameters] = None, rng: Optional[np.random.Generator] = None):
        self.params = params or LightCurveParameters()
        self.rng = rng or np.random.default_rng()

    def generate_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        widths = self.rng.normal(self.params.mean_bin_width, self.params.std_bin_width, self.params.n_bins)
        widths = np.clip(widths, 1.0, 100.0)
        cumsum = np.cumsum(widths)
        t = cumsum - widths / 2.0
        return t, widths

    def generate_blocks(self, t: np.ndarray) -> np.ndarray:
        blocks = [0]
        for i in range(1, len(t)):
            if self.rng.random() < self.params.prob_new_block:
                blocks.append(i)
        if blocks[-1] != len(t) - 1:
            blocks.append(len(t) - 1)
        return np.array(blocks, dtype=int)

    def generate_flux_values(self, block_indices: np.ndarray, flux_params: Optional[FluxParameters] = None) -> np.ndarray:
        flux_params = flux_params or FluxParameters()
        n_bins = self.params.n_bins
        n_blocks = len(block_indices)
        # draw a per-block flux value (lognormal centered near mean_flux)
        block_fluxes = self.rng.lognormal(np.log(max(flux_params.min_flux, flux_params.mean_flux)), flux_params.flux_std, n_blocks)
        block_fluxes = np.clip(block_fluxes, flux_params.min_flux, flux_params.max_flux)

        # assign block fluxes to bins
        flux = np.empty(n_bins, dtype=float)
        for i, start in enumerate(block_indices):
            # end is either next block start or end of array
            end = block_indices[i + 1] if (i + 1) < len(block_indices) else n_bins
            flux[start:end] = block_fluxes[i]

        # add bin-level multiplicative noise
        flux *= self.rng.normal(1.0, 0.1, n_bins)
        flux = np.clip(flux, flux_params.min_flux, flux_params.max_flux)
        return flux

    def generate_errors(self, flux: np.ndarray) -> np.ndarray:
        # errors scale with flux
        relative_error = self.rng.uniform(0.05, 0.2, len(flux))
        sigma = flux * relative_error
        # ensure a small floor
        sigma = np.clip(sigma, 1e-6, None)
        return sigma

    def generate(self, flux_params: Optional[FluxParameters] = None) -> Dict[str, Any]:
        t, tw = self.generate_bins()
        block_indices = self.generate_blocks(t)
        flux = self.generate_flux_values(block_indices, flux_params)
        errors = self.generate_errors(flux)
        return {
            "t": t.tolist(),
            "tw": tw.tolist(),
            "flux": flux.tolist(),
            "errors": errors.tolist(),
            "block_indices": block_indices.tolist()
        }


class SourcePropertyGenerator:
    ASSOCIATIONS = ["bll", "fsrq", "bcu", "psr", "unid", "other"]

    @staticmethod
    def generate_coordinates(n_sources: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        glon = rng.uniform(0.0, 360.0, n_sources)
        glat = rng.uniform(-90.0, 90.0, n_sources)
        return glon, glat

    @staticmethod
    def generate_test_statistic(n_sources: int, rng: np.random.Generator) -> np.ndarray:
        ts = rng.lognormal(3.5, 1.5, n_sources)
        return np.clip(ts, 25.0, 2000.0)

    @staticmethod
    def generate_eflux100(n_sources: int, rng: np.random.Generator) -> np.ndarray:
        log_eflux = rng.uniform(-13.0, -11.0, n_sources)
        return 10.0 ** log_eflux

    @staticmethod
    def generate_spectral_index(n_sources: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(2.0, 0.4, n_sources).clip(1.0, 3.0)

    @staticmethod
    def generate_variability_index(n_sources: int, rng: np.random.Generator) -> np.ndarray:
        is_variable = rng.random(n_sources) < 0.1
        bbvar = np.where(is_variable, rng.lognormal(2.0, 1.5, n_sources), rng.uniform(0.5, 3.0, n_sources))
        return np.clip(bbvar, 0.1, 200.0)

    @staticmethod
    def generate_associations(n_sources: int, rng: np.random.Generator) -> List[str]:
        weights = [0.35, 0.20, 0.10, 0.05, 0.20, 0.10]
        return rng.choice(SourcePropertyGenerator.ASSOCIATIONS, size=n_sources, p=weights).tolist()


class SimulatedSourceDatabase:
    """
    Simulated source catalog + light curves.

    - stores 'sources' as OrderedDict keyed by source id strings
    - provides to_dict(), to_dataframe(), save(), etc.
    """

    def __init__(self, n_sources: int = 100, seed: Optional[int] = None):
        self.n_sources = n_sources
        self.seed = seed
        self.sources: "OrderedDict[str, dict]" = OrderedDict()
        self.metadata: Dict[str, Any] = {}
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def generate(self,
                 lc_params: Optional[LightCurveParameters] = None,
                 flux_params: Optional[FluxParameters] = None,
                 verbose: bool = True) -> "SimulatedSourceDatabase":
        lc_params = lc_params or LightCurveParameters(n_bins=self.n_sources and 100)
        flux_params = flux_params or FluxParameters()

        if verbose:
            print(f"Generating {self.n_sources} simulated sources (seed={self.seed})...")

        glon, glat = SourcePropertyGenerator.generate_coordinates(self.n_sources, self._rng)
        ts = SourcePropertyGenerator.generate_test_statistic(self.n_sources, self._rng)
        eflux100 = SourcePropertyGenerator.generate_eflux100(self.n_sources, self._rng)
        pindex = SourcePropertyGenerator.generate_spectral_index(self.n_sources, self._rng)
        bbvar = SourcePropertyGenerator.generate_variability_index(self.n_sources, self._rng)
        associations = SourcePropertyGenerator.generate_associations(self.n_sources, self._rng)

        for i in range(self.n_sources):
            source_id = f"4FGL J{i:04d}"
            # per-source RNG to make per-source results independent but reproducible
            per_source_seed = (self.seed or 0) + i * 1007
            per_rng = np.random.default_rng(per_source_seed)

            lc_gen = SimulatedLightCurve(params=lc_params, rng=per_rng)
            light_curve = lc_gen.generate(flux_params)

            # nearby sources (simple)
            n_nearby = per_rng.integers(0, 5)
            nearby = [{"jname": f"nearby_{j}", "sep": float(per_rng.uniform(0.1, 5.0))} for j in range(n_nearby)]
            nearby = nearby if nearby else None

            poisson_fits = [(float(1.0 + per_rng.normal(0, 0.1)), float(per_rng.uniform(0.05, 0.15)))
                            for _ in range(len(light_curve["flux"]))]

            self.sources[source_id] = {
                "light_curve": light_curve,
                "nearby": nearby,
                "poisson_fits": poisson_fits,
                "ts": float(ts[i]),
                "eflux100": float(eflux100[i]),
                "pindex": float(pindex[i]),
                "bbvar": float(bbvar[i]),
                "variability": int(per_rng.integers(1, 100)),
                "association": associations[i],
                "glon": float(glon[i]),
                "glat": float(glat[i]),
            }

            if verbose and (i + 1) % max(1, self.n_sources // 10) == 0:
                print(f"  Generated {i + 1}/{self.n_sources} sources")

        self.metadata = {
            "n_sources": self.n_sources,
            "seed": self.seed,
            "lc_params": asdict(lc_params),
            "flux_params": asdict(flux_params),
        }
        return self

    def to_dict(self) -> Dict[str, dict]:
        return dict(self.sources)

    def save(self, filepath: str) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.to_dict(), f)
        print(f"Saved simulated database to {filepath}")

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for source_id, data in self.sources.items():
            rows.append({
                "source_id": source_id,
                "ts": data["ts"],
                "eflux100": data["eflux100"],
                "pindex": data["pindex"],
                "bbvar": data["bbvar"],
                "variability": data["variability"],
                "association": data["association"],
                "glon": data["glon"],
                "glat": data["glat"],
                "nbins": len(data["light_curve"]["flux"]) if data["light_curve"] else 0,
                "nblocks": len(data["light_curve"].get("block_indices", [])),
                "near": len(data["nearby"]) if data["nearby"] else 0,
            })
        return pd.DataFrame(rows)


# -------------------------
# Bayesian block detection
# -------------------------
class BayesianBlockCounter:
    """
    Lightweight block counter approximating a BB-based change point detector.
    Two methods:
      - count_blocks_stochastic: random-process based count
      - count_blocks_from_lightcurve: z-score-based local change detection
    """

    @staticmethod
    def count_blocks_stochastic(n_bins: int, prob_new_block: float, rng: Optional[np.random.Generator] = None) -> int:
        rng = rng or np.random.default_rng()
        blocks = [0]
        for i in range(1, n_bins):
            if rng.random() < prob_new_block:
                blocks.append(i)
        if blocks[-1] != n_bins - 1:
            blocks.append(n_bins - 1)
        return len(blocks)

    @staticmethod
    def count_blocks_from_lightcurve(flux: np.ndarray, errors: Optional[np.ndarray] = None, p0: float = 0.05) -> int:
        flux = np.asarray(flux, dtype=float)
        n_bins = len(flux)
        if errors is None:
            errors = np.full(n_bins, 0.1)
        else:
            errors = np.asarray(errors, dtype=float)
            if errors.shape != flux.shape:
                raise ValueError("errors must have the same shape as flux")

        blocks = [0]
        z_thresh = stats.norm.ppf(1.0 - p0 / 2.0)
        for i in range(1, n_bins):
            flux_change = abs(flux[i] - flux[i - 1])
            combined_error = np.sqrt(errors[i] ** 2 + errors[i - 1] ** 2)
            if combined_error > 0:
                z_score = flux_change / combined_error
                if z_score > z_thresh:
                    blocks.append(i)
        if blocks[-1] != n_bins - 1:
            blocks.append(n_bins - 1)
        return len(blocks)


# -------------------------
# Monte Carlo simulations
# -------------------------
class MonteCarloBlockSimulation:
    """
    Run MC simulations for Bayesian-block counts.

    Usage patterns:
    - instantiate with BBCountSimulationConfig
    - optionally supply a SimulatedSourceDatabase to use its light curves
    - call run_null_hypothesis_simulation() or run_variable_source_simulation()
    - get results as pandas DataFrame via self.detailed_results
    """

    def __init__(self, config: Optional[BBCountSimulationConfig] = None, rng: Optional[np.random.Generator] = None):
        self.config = config or BBCountSimulationConfig()
        self.rng = rng or np.random.default_rng(self.config.seed_base)
        self.detailed_results: Optional[pd.DataFrame] = None

    def _simulate_lightcurve_constant(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        n_bins = self.config.n_bins_per_lc
        level = self.config.constant_flux_level
        # Poisson draw scaled to retain a flux-like quantity (avoid zero errors)
        raw = rng.poisson(max(1, int(level)), n_bins).astype(float)
        # normalize such that mean ~ constant flux level
        flux = (raw / max(1.0, raw.mean())) * level
        errors = np.sqrt(raw) / max(1.0, raw.mean())
        errors = np.clip(errors, 1e-6, None)
        return flux, errors

    def run_null_hypothesis_simulation(self, use_sim_db: Optional[SimulatedSourceDatabase] = None) -> pd.DataFrame:
        results = []
        n_sim = self.config.n_simulations
        print(f"Running {n_sim} null-hypothesis simulations (seed_base={self.config.seed_base})")
        for sim_id in range(n_sim):
            sim_rng = np.random.default_rng(self.config.seed_base + sim_id)
            # If a precomputed database is provided, draw light curves from it
            if use_sim_db is not None:
                # sample sources from the database (with replacement if needed)
                db_sources = list(use_sim_db.sources.items())
                for source_idx in range(self.config.n_sources):
                    sid, sdata = db_sources[source_idx % len(db_sources)]
                    flux = np.asarray(sdata["light_curve"]["flux"], dtype=float)
                    errors = np.asarray(sdata["light_curve"]["errors"], dtype=float)
                    n_blocks = BayesianBlockCounter.count_blocks_from_lightcurve(flux, errors)
                    results.append({
                        "simulation_id": sim_id,
                        "source_id": sid,
                        "n_blocks": n_blocks,
                        "flux_level": float(np.mean(flux)),
                        "flux_std": float(np.std(flux))
                    })
            else:
                for source_id in range(self.config.n_sources):
                    if self.config.use_constant_flux:
                        flux, errors = self._simulate_lightcurve_constant(sim_rng)
                    else:
                        # random constant level per source
                        level = sim_rng.uniform(0.5, 2.0)
                        flux = sim_rng.normal(level, level * 0.1, self.config.n_bins_per_lc)
                        flux = np.clip(flux, 0.01, 10.0)
                        errors = np.abs(flux) * 0.1

                    n_blocks = BayesianBlockCounter.count_blocks_from_lightcurve(flux, errors)
                    results.append({
                        "simulation_id": sim_id,
                        "source_id": source_id,
                        "n_blocks": n_blocks,
                        "flux_level": float(np.mean(flux)),
                        "flux_std": float(np.std(flux))
                    })
            if (sim_id + 1) % max(1, n_sim // 10) == 0:
                print(f"  Completed {sim_id + 1}/{n_sim}")
        self.detailed_results = pd.DataFrame(results)
        return self.detailed_results

    def run_variable_source_simulation(self,
                                       n_steps: Optional[List[int]] = None,
                                       step_sizes: Optional[List[float]] = None) -> pd.DataFrame:
        n_steps = n_steps or [1, 2, 3]
        step_sizes = step_sizes or [1.5, 2.0]
        results = []
        n_sim = self.config.n_simulations
        for sim_id in range(n_sim):
            sim_rng = np.random.default_rng(self.config.seed_base + sim_id)
            for source_id in range(self.config.n_sources):
                # pick number of injected steps by cycling through choices
                n_inject = n_steps[source_id % len(n_steps)]
                for step_size in step_sizes:
                    n_bins = self.config.n_bins_per_lc
                    bins_per_block = max(1, n_bins // (n_inject + 1))
                    # make a stepped light curve (levels multiply by step_size at each step)
                    flux = np.ones(n_bins)
                    current = 1.0
                    for step_idx in range(n_inject):
                        start = (step_idx + 1) * bins_per_block
                        current *= step_size
                        flux[start:] *= current
                    # add noise
                    flux = flux * sim_rng.normal(1.0, 0.1, n_bins)
                    flux = np.clip(flux, 0.01, 10.0)
                    errors = np.abs(flux) * 0.1
                    n_detected = BayesianBlockCounter.count_blocks_from_lightcurve(flux, errors)
                    results.append({
                        "simulation_id": sim_id,
                        "source_id": source_id,
                        "step_size": step_size,
                        "n_injected_blocks": n_inject + 1,
                        "n_detected_blocks": n_detected,
                        "detection_efficiency": 1.0 if n_detected == (n_inject + 1) else 0.0,
                        "n_bins_per_block": bins_per_block,
                        "mean_flux_ratio": float(np.max(flux) / np.min(flux))
                    })
            if (sim_id + 1) % max(1, n_sim // 10) == 0:
                print(f"  Completed {sim_id + 1}/{n_sim}")
        self.detailed_results = pd.DataFrame(results)
        return self.detailed_results

    def summarize_results(self) -> Dict[str, Any]:
        if self.detailed_results is None:
            raise ValueError("No results available. Run a simulation first.")
        df = self.detailed_results
        summary = {
            "n_total_rows": len(df),
            "n_simulations": int(df["simulation_id"].nunique()),
            "sources_per_sim": int(len(df) // max(1, df["simulation_id"].nunique()))
        }
        if "n_blocks" in df.columns:
            summary.update({
                "mean_n_blocks": float(df["n_blocks"].mean()),
                "median_n_blocks": float(df["n_blocks"].median()),
                "std_n_blocks": float(df["n_blocks"].std()),
                "min_n_blocks": int(df["n_blocks"].min()),
                "max_n_blocks": int(df["n_blocks"].max()),
                "frac_single_block": float((df["n_blocks"] == 1).sum() / len(df)),
            })
            summary["block_distribution"] = df["n_blocks"].value_counts().to_dict()
        if "detection_efficiency" in df.columns:
            summary.update({
                "mean_detection_efficiency": float(df["detection_efficiency"].mean()),
                "detection_efficiency_by_step": df.groupby("step_size")["detection_efficiency"].mean().to_dict()
            })
        return summary


# -------------------------
# Visualization utilities
# -------------------------
class BlockCountVisualizer:
    @staticmethod
    def plot_null_hypothesis_distribution(sim: MonteCarloBlockSimulation, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        if sim.detailed_results is None:
            raise ValueError("Simulation has no detailed results.")
        df = sim.detailed_results
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        # histogram of counts
        ax = axes[0]
        ax.hist(df["n_blocks"], bins=np.arange(0.5, df["n_blocks"].max() + 1.5), color="C0", histtype="stepfilled", alpha=0.6)
        ax.set_xlabel("Number of Bayesian Blocks")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Block Counts (Null Hypothesis)")

        # mean blocks per simulation
        ax = axes[1]
        sim_means = df.groupby("simulation_id")["n_blocks"].mean()
        ax.hist(sim_means, bins=20, color="C1", alpha=0.7)
        ax.set_xlabel("Mean Blocks per Simulation")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Mean Blocks Across Simulations")

        # CDF
        ax = axes[2]
        sorted_blocks = np.sort(df["n_blocks"].values)
        cdf = np.arange(1, len(sorted_blocks) + 1) / len(sorted_blocks)
        ax.step(sorted_blocks, cdf, where="post", color="C0")
        ax.set_xlabel("Number of Bayesian Blocks")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("CDF of Block Counts")

        # boxplots by first few sims
        ax = axes[3]
        sims = sorted(df["simulation_id"].unique())[:5]
        data_by_sim = [df[df["simulation_id"] == s]["n_blocks"].values for s in sims]
        if any(len(x) == 0 for x in data_by_sim):
            ax.text(0.5, 0.5, "Not enough data for boxplots", ha="center")
        else:
            bp = ax.boxplot(data_by_sim, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("C0")
        ax.set_title("Block Count Distribution (First 5 Simulations)")

        # flux vs blocks
        ax = axes[4]
        ax.scatter(df["flux_level"], df["n_blocks"], s=8, alpha=0.6, c=df.get("flux_std", None), cmap="viridis")
        ax.set_xlabel("Mean Flux Level")
        ax.set_ylabel("Number of Blocks")
        ax.set_title("Flux Level vs Block Count")

        # fraction by block count
        ax = axes[5]
        block_counts = df["n_blocks"].value_counts().sort_index()
        total = len(df)
        fractions = block_counts / total
        ax.bar(fractions.index, fractions.values, color="C0", alpha=0.7)
        ax.set_xlabel("Number of Blocks")
        ax.set_ylabel("Fraction of Sources")
        ax.set_title("Fraction Distribution of Block Counts")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_detection_efficiency(sim: MonteCarloBlockSimulation, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        if sim.detailed_results is None:
            raise ValueError("Simulation has no detailed results.")
        df = sim.detailed_results
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        ax = axes[0]
        eff_by_step = df.groupby("step_size")["detection_efficiency"].mean()
        eff_by_step.plot(kind="line", marker="o", ax=ax, color="C2")
        ax.set_xlabel("Injected Step Size (Flux Ratio)")
        ax.set_ylabel("Detection Efficiency")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Ability to Detect Injected Steps")

        ax = axes[1]
        scatter = ax.scatter(df["n_injected_blocks"], df["n_detected_blocks"], c=df["step_size"], cmap="viridis", alpha=0.6)
        max_blocks = max(df["n_injected_blocks"].max(), df["n_detected_blocks"].max())
        ax.plot([0, max_blocks], [0, max_blocks], "r--", label="Perfect detection")
        ax.set_xlabel("Injected Blocks")
        ax.set_ylabel("Detected Blocks")
        ax.set_title("Detected vs Injected Block Counts")
        plt.colorbar(scatter, ax=ax, label="Step Size")
        ax.legend()
        plt.tight_layout()
        return fig


# -------------------------
# Simplified top-level API
# -------------------------
def generate_simulated_db(n_sources: int = 100, seed: Optional[int] = None,
                          lc_params: Optional[LightCurveParameters] = None,
                          flux_params: Optional[FluxParameters] = None,
                          verbose: bool = True) -> SimulatedSourceDatabase:
    db = SimulatedSourceDatabase(n_sources=n_sources, seed=seed)
    db.generate(lc_params=lc_params, flux_params=flux_params, verbose=verbose)
    return db


def run_null_simulation(config: Optional[BBCountSimulationConfig] = None,
                        use_sim_db: Optional[SimulatedSourceDatabase] = None) -> Tuple[MonteCarloBlockSimulation, pd.DataFrame]:
    mc = MonteCarloBlockSimulation(config=config)
    df = mc.run_null_hypothesis_simulation(use_sim_db=use_sim_db)
    return mc, df


def run_injected_variability(config: Optional[BBCountSimulationConfig] = None,
                             n_steps: Optional[List[int]] = None,
                             step_sizes: Optional[List[float]] = None) -> Tuple[MonteCarloBlockSimulation, pd.DataFrame]:
    mc = MonteCarloBlockSimulation(config=config)
    df = mc.run_variable_source_simulation(n_steps=n_steps, step_sizes=step_sizes)
    return mc, df


# -------------------------
# CLI / example usage
# -------------------------
if __name__ == "__main__":
    print("Combined Simulation module - example run")
    print("=" * 70)

    # Example: generate a small simulated database
    db = generate_simulated_db(n_sources=50, seed=42, verbose=True)
    df_db = db.to_dataframe()
    print("\nSimulated DB Summary:")
    print(df_db.describe(include="all"))

    # Example: run a small null-hypothesis MC
    cfg = BBCountSimulationConfig(n_simulations=5, n_sources=50, n_bins_per_lc=100, seed_base=100)
    mc, df_null = run_null_simulation(config=cfg)
    print("\nNull-hypothesis simulation summary:")
    print(mc.summarize_results())

    # Save null results
    out_dir = Path("./bb_mc_results_example")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_null.to_csv(out_dir / "null_results_example.csv", index=False)
    print(f"Saved results to {out_dir / 'null_results_example.csv'}")

    # Example: run injected-variability simulation
    mc2, df_var = run_injected_variability(config=BBCountSimulationConfig(n_simulations=3, n_sources=40, n_bins_per_lc=100, seed_base=200),
                                           n_steps=[1, 2], step_sizes=[1.5, 2.5])
    print("\nInjected-variability summary:")
    print(mc2.summarize_results())

    # Example plotting (will open figures; save to disk)
    fig1 = BlockCountVisualizer.plot_null_hypothesis_distribution(mc)
    fig1.savefig(out_dir / "null_distribution_example.png", dpi=150)
    fig2 = BlockCountVisualizer.plot_detection_efficiency(mc2)
    fig2.savefig(out_dir / "detection_efficiency_example.png", dpi=150)
    print(f"Saved figures to {out_dir}")