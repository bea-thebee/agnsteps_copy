"""
Analyze a single wtlike source by repeated resampling of its light curve
and running a Bayesian-blocks-like detector on each resample.

Usage (example):
    python analyze_wtlike_source.py \
        --input ./wtlike_outputs/sim_realization.pkl \
        --source 4FGL_J0001 \
        --n_trials 1000 \
        --method sample_errors \
        --out results_source_4FGL_J0001.csv

The script returns a CSV (if --out passed) and prints a summary of detected
block-count frequencies.
"""
from typing import Optional, Tuple, Dict, Any
import argparse
from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# Try to import astropy's bayesian_blocks if available (optional)
try:
    from astropy.stats import bayesian_blocks as astropy_bayesian_blocks  # type: ignore
except Exception:
    astropy_bayesian_blocks = None


def _load_file(path: Path) -> Any:
    """Load a file heuristically (pickle, npz, json, csv, hdf)."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    if path.suffix.lower() == ".npz":
        return dict(np.load(str(path), allow_pickle=True))
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            return json.load(f)
    if path.suffix.lower() in (".csv",):
        return pd.read_csv(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        return pd.read_hdf(path)
    # fallback: attempt pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def find_source_lightcurve(loaded_obj: Any, source_key: str) -> Dict[str, Any]:
    """
    Given a loaded object and a source key/name, try to extract a light curve
    dict with keys: 'flux' (list/array), optionally 'errors', 't', 'tw', 'block_indices'.
    """
    # If DataFrame with source_id
    if isinstance(loaded_obj, pd.DataFrame):
        df = loaded_obj
        if "source_id" in df.columns:
            grp = df[df["source_id"].astype(str) == str(source_key)]
            if grp.empty:
                raise KeyError(f"source {source_key} not found in DataFrame")
            # assume 'flux' and optionally 'flux_err' or 'errors'
            flux_col = "flux" if "flux" in grp.columns else grp.columns[0]
            errs_col = "flux_err" if "flux_err" in grp.columns else ("errors" if "errors" in grp.columns else None)
            return {"flux": grp[flux_col].to_numpy(), "errors": grp[errs_col].to_numpy() if errs_col else None}
        else:
            # treat as single timeseries
            flux_col = "flux" if "flux" in df.columns else df.columns[0]
            errs_col = "flux_err" if "flux_err" in df.columns else ("errors" if "errors" in df.columns else None)
            return {"flux": df[flux_col].to_numpy(), "errors": df[errs_col].to_numpy() if errs_col else None}

    # If dict-like (common for pickles or npz)
    if isinstance(loaded_obj, dict):
        # direct key match
        if source_key in loaded_obj:
            v = loaded_obj[source_key]
            # v might be a dict with light_curve key
            if isinstance(v, dict):
                if "light_curve" in v and isinstance(v["light_curve"], dict):
                    lc = v["light_curve"]
                    return {"flux": np.asarray(lc.get("flux", [])), "errors": np.asarray(lc.get("errors")) if lc.get("errors") is not None else None, "t": lc.get("t"), "tw": lc.get("tw")}
                if "flux" in v:
                    return {"flux": np.asarray(v.get("flux")), "errors": np.asarray(v.get("errors")) if v.get("errors") is not None else None}
            # if v is array-like treat as flux vector
            if isinstance(v, (list, tuple, np.ndarray)):
                return {"flux": np.asarray(v), "errors": None}
        # try to find a top-level 'sources' key or similar
        for candidate in ("sources", "simulated_sources", "db", "catalog"):
            if candidate in loaded_obj and isinstance(loaded_obj[candidate], dict):
                if source_key in loaded_obj[candidate]:
                    v = loaded_obj[candidate][source_key]
                    if isinstance(v, dict) and "light_curve" in v:
                        lc = v["light_curve"]
                        return {"flux": np.asarray(lc.get("flux", [])), "errors": np.asarray(lc.get("errors")) if lc.get("errors") is not None else None}
        # fallback: if dict maps many source-like keys to light curves,
        # try approximate key matching (replace spaces, dots, etc.)
        sk = str(source_key).replace(" ", "").replace(".", "").lower()
        for k, v in loaded_obj.items():
            if not isinstance(k, str):
                continue
            if k.replace(" ", "").replace(".", "").lower() == sk:
                if isinstance(v, dict) and "light_curve" in v:
                    lc = v["light_curve"]
                    return {"flux": np.asarray(lc.get("flux", [])), "errors": np.asarray(lc.get("errors")) if lc.get("errors") is not None else None}
                if isinstance(v, (list, np.ndarray)):
                    return {"flux": np.asarray(v), "errors": None}
        raise KeyError(f"source {source_key} not found in dict-like file. Available keys (sample): {list(loaded_obj.keys())[:20]}")

    # if object is a list/ndarray, assume it's a flux vector
    if isinstance(loaded_obj, (list, tuple, np.ndarray)):
        return {"flux": np.asarray(loaded_obj), "errors": None}

    # otherwise cannot interpret
    raise TypeError("Unsupported file structure for extracting a source light curve")


def count_blocks_from_lightcurve(flux: np.ndarray, errors: Optional[np.ndarray] = None, p0: float = 0.05, use_astropy: bool = False) -> int:
    """
    Simplified BB detector: either use astropy.stats.bayesian_blocks (if available and requested)
    or the z-score based local-change detector used in the merged module.
    """
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    if n == 0:
        return 0
    if use_astropy and astropy_bayesian_blocks is not None:
        # astropy's bayesian_blocks expects event times or a time series; use indices as times
        edges = astropy_bayesian_blocks(np.arange(n), flux)
        # edges are change-point boundaries; number of blocks = len(edges)-1
        return max(1, len(edges) - 1)
    # z-score approach
    if errors is None:
        errors = np.full(n, 0.1)
    else:
        errors = np.asarray(errors, dtype=float)
        if errors.shape != flux.shape:
            # try to coerce or fallback to relative errors
            errors = np.full(n, np.std(flux) * 0.1 if np.std(flux) > 0 else 0.1)
    blocks = [0]
    z_thresh = stats.norm.ppf(1.0 - p0 / 2.0)
    for i in range(1, n):
        change = abs(flux[i] - flux[i - 1])
        comb_err = np.sqrt(errors[i] ** 2 + errors[i - 1] ** 2)
        if comb_err > 0:
            z = change / comb_err
            if z > z_thresh:
                blocks.append(i)
    if blocks[-1] != n - 1:
        blocks.append(n - 1)
    return len(blocks)


def analyze_source_repeatedly(input_path: str, source_key: str, n_trials: int = 1000,
                              method: str = "sample_errors", jitter_scale: float = 0.1, p0: float = 0.05,
                              use_astropy: bool = False, seed: Optional[int] = None,
                              out_csv: Optional[str] = None, plot_hist: Optional[str] = None) -> pd.DataFrame:
    """
    Main function:
      - loads the file at input_path
      - extracts the source light curve with key source_key
      - runs n_trials resamples according to method
      - runs the BB detector on each trial
      - returns DataFrame of results
    """
    path = Path(input_path)
    loaded = _load_file(path)
    lc_info = find_source_lightcurve(loaded, source_key)
    flux0 = np.asarray(lc_info.get("flux", []), dtype=float)
    errors0 = lc_info.get("errors", None)
    if errors0 is not None:
        errors0 = np.asarray(errors0, dtype=float)
        if errors0.shape != flux0.shape:
            # if lengths mismatch, ignore errors
            errors0 = None

    rng = np.random.default_rng(seed)

    records = []
    for t in range(n_trials):
        # use different RNG state per trial for reproducibility if seed provided
        if method == "sample_errors" and errors0 is not None:
            flux_sim = rng.normal(loc=flux0, scale=errors0)
            # ensure positive fluxes
            flux_sim = np.clip(flux_sim, a_min=0.0, a_max=None)
            errors_sim = errors0.copy()
        elif method == "jitter_multiplicative":
            flux_sim = flux0 * rng.normal(1.0, jitter_scale, size=flux0.shape)
            flux_sim = np.clip(flux_sim, 0.0, None)
            errors_sim = np.abs(flux_sim) * 0.1
        elif method == "sample_errors_with_floor":
            if errors0 is not None:
                floor = np.maximum(errors0, jitter_scale * np.maximum(1.0, flux0))
                flux_sim = rng.normal(flux0, floor)
                flux_sim = np.clip(flux_sim, 0.0, None)
                errors_sim = floor
            else:
                flux_sim = flux0 * rng.normal(1.0, jitter_scale, size=flux0.shape)
                flux_sim = np.clip(flux_sim, 0.0, None)
                errors_sim = np.abs(flux_sim) * 0.1
        else:
            # default fallback - multiplicative jitter
            flux_sim = flux0 * rng.normal(1.0, jitter_scale, size=flux0.shape)
            flux_sim = np.clip(flux_sim, 0.0, None)
            errors_sim = np.abs(flux_sim) * 0.1

        n_blocks = count_blocks_from_lightcurve(flux_sim, errors_sim, p0=p0, use_astropy=use_astropy)
        records.append({
            "trial": t,
            "n_blocks": int(n_blocks),
            "mean_flux": float(np.mean(flux_sim)) if flux_sim.size else 0.0,
            "std_flux": float(np.std(flux_sim)) if flux_sim.size else 0.0,
        })

    df = pd.DataFrame.from_records(records)
    # summary
    counts = df["n_blocks"].value_counts().sort_index()
    total = len(df)
    summary = (counts / total).to_dict()

    print(f"Analysis for source '{source_key}' in {input_path}:")
    print(f"  Trials: {n_trials}")
    print("  Block count distribution (counts):")
    print(counts.to_string())
    print("  Block count distribution (fractions):")
    for k, v in summary.items():
        print(f"    {k} blocks: {v:.3%}")

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Saved trial-level results to {out_csv}")

    if plot_hist:
        plt.figure(figsize=(6, 4))
        plt.hist(df["n_blocks"], bins=range(int(df["n_blocks"].max()) + 2), align="left", rwidth=0.8, color="C0", alpha=0.8)
        plt.xlabel("Number of detected blocks")
        plt.ylabel("Count")
        plt.title(f"Block counts over {n_trials} trials for {source_key}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_hist, dpi=150)
        plt.close()
        print(f"Saved histogram to {plot_hist}")

    return df


def _parse_args():
    p = argparse.ArgumentParser(description="Resample a wtlike source light curve and run BB detection repeatedly.")
    p.add_argument("--input", "-i", required=True, help="Path to wtlike simulation file (pkl/npz/csv/h5/json)")
    p.add_argument("--source", "-s", required=True, help="Source key / source_id to extract")
    p.add_argument("--n_trials", "-n", type=int, default=1000, help="Number of resampling trials")
    p.add_argument("--method", "-m", choices=["sample_errors", "jitter_multiplicative", "sample_errors_with_floor"], default="sample_errors", help="Resampling method")
    p.add_argument("--jitter_scale", type=float, default=0.1, help="Multiplicative jitter sigma (for jitter methods)")
    p.add_argument("--p0", type=float, default=0.05, help="Significance threshold for z-score BB detector")
    p.add_argument("--use_astropy", action="store_true", help="Use astropy.stats.bayesian_blocks if available")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--out", type=str, default=None, help="CSV path to save trial-level results")
    p.add_argument("--hist", type=str, default=None, help="PNG path to save histogram of block counts")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df_result = analyze_source_repeatedly(
        input_path=args.input,
        source_key=args.source,
        n_trials=args.n_trials,
        method=args.method,
        jitter_scale=args.jitter_scale,
        p0=args.p0,
        use_astropy=args.use_astropy,
        seed=args.seed,
        out_csv=args.out,
        plot_hist=args.hist
    )