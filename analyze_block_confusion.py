#!/usr/bin/env python3
"""
analyze_block_confusion.py

Resample a wtlike source light curve with injected steps and analyze
margin-of-error and confusion over block sizes.

Features
- Load a single source light curve from common file formats (pkl/npz/csv/h5/json).
- Inject S steps (S+1 true blocks) at randomized positions/durations.
- Run a simplified Bayesian-blocks-like detector (z-score) many times.
- Compute confusion matrix (true blocks vs detected blocks).
- Compute detection efficiency with 95% Wilson confidence intervals.
- Analyze detection dependence on bins-per-block (block size).
- Save trial-level CSV and summary plots.

Usage examples:
  python analyze_block_confusion.py --input ./wtlike_sim.pkl --source 4FGL_J0001 --n_trials 2000 --step_sizes 1.5 2.0 --n_steps 1 2
  python analyze_block_confusion.py --n_trials 1000 --synthetic --n_bins 100 --source_id synthetic_01

Author: assistant (combined/adapted)
"""
from pathlib import Path
import argparse
import pickle
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Optional astropy Bayesian Blocks (if you want to use it; not required)
try:
    from astropy.stats import bayesian_blocks as astropy_bayesian_blocks  # type: ignore
except Exception:
    astropy_bayesian_blocks = None


# ---------- Utilities ----------

def _load_file(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf in ('.pkl', '.pickle'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    if suf == '.npz':
        return dict(np.load(path, allow_pickle=True))
    if suf in ('.json',):
        with open(path, 'r') as f:
            return json.load(f)
    if suf in ('.csv',):
        return pd.read_csv(path)
    if suf in ('.h5', '.hdf5'):
        return pd.read_hdf(path)
    # fallback: try pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def find_source_lightcurve(obj: Any, source_key: str) -> Dict[str, Any]:
    """
    Attempt to extract light curve dict with 'flux' and optionally 'errors' from loaded object.
    """
    # DataFrame with source_id
    if isinstance(obj, pd.DataFrame):
        df = obj
        if 'source_id' in df.columns:
            grp = df[df['source_id'].astype(str) == str(source_key)]
            if grp.empty:
                raise KeyError(f"source {source_key} not found in DataFrame")
            flux_col = 'flux' if 'flux' in grp.columns else grp.columns[0]
            errs_col = 'flux_err' if 'flux_err' in grp.columns else ('errors' if 'errors' in grp.columns else None)
            return {'flux': grp[flux_col].to_numpy(), 'errors': grp[errs_col].to_numpy() if errs_col else None}
        else:
            flux_col = 'flux' if 'flux' in df.columns else df.columns[0]
            errs_col = 'flux_err' if 'flux_err' in df.columns else ('errors' if 'errors' in df.columns else None)
            return {'flux': df[flux_col].to_numpy(), 'errors': df[errs_col].to_numpy() if errs_col else None}

    if isinstance(obj, dict):
        # direct match
        if source_key in obj:
            v = obj[source_key]
            if isinstance(v, dict):
                if 'light_curve' in v and isinstance(v['light_curve'], dict):
                    lc = v['light_curve']
                    return {'flux': np.asarray(lc.get('flux', [])), 'errors': np.asarray(lc.get('errors')) if lc.get('errors') is not None else None}
                if 'flux' in v:
                    return {'flux': np.asarray(v.get('flux')), 'errors': np.asarray(v.get('errors')) if v.get('errors') is not None else None}
            if isinstance(v, (list, tuple, np.ndarray)):
                return {'flux': np.asarray(v), 'errors': None}
        # common container keys
        for candidate in ('sources', 'simulated_sources', 'db', 'catalog'):
            if candidate in obj and isinstance(obj[candidate], dict):
                if source_key in obj[candidate]:
                    v = obj[candidate][source_key]
                    if isinstance(v, dict) and 'light_curve' in v:
                        lc = v['light_curve']
                        return {'flux': np.asarray(lc.get('flux', [])), 'errors': np.asarray(lc.get('errors')) if lc.get('errors') is not None else None}
        # approximate string matching fallback
        sk = str(source_key).replace(' ', '').replace('.', '').lower()
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if k.replace(' ', '').replace('.', '').lower() == sk:
                if isinstance(v, dict) and 'light_curve' in v:
                    lc = v['light_curve']
                    return {'flux': np.asarray(lc.get('flux', [])), 'errors': np.asarray(lc.get('errors')) if lc.get('errors') is not None else None}
                if isinstance(v, (list, np.ndarray)):
                    return {'flux': np.asarray(v), 'errors': None}
        raise KeyError(f"source {source_key} not found. Available keys: {list(obj.keys())[:30]}")

    if isinstance(obj, (list, tuple, np.ndarray)):
        return {'flux': np.asarray(obj), 'errors': None}

    raise TypeError("Unsupported file structure for extracting light curve")


# Wilson score interval for a proportion
def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# simplified z-score local change detector as before
def detect_blocks_zscore(flux: np.ndarray, errors: Optional[np.ndarray] = None, p0: float = 0.05) -> int:
    flux = np.asarray(flux, dtype=float)
    n = len(flux)
    if n == 0:
        return 0
    if errors is None:
        errors = np.full(n, 0.1)
    else:
        errors = np.asarray(errors, dtype=float)
        if errors.shape != flux.shape:
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


def detect_blocks_astropy(flux: np.ndarray) -> int:
    if astropy_bayesian_blocks is None:
        raise RuntimeError("astropy.stats.bayesian_blocks not available")
    n = len(flux)
    if n == 0:
        return 0
    edges = astropy_bayesian_blocks(np.arange(n), flux)
    return max(1, len(edges) - 1)


# ---------- Injection and experiment ----------

def inject_steps_random(base_flux: np.ndarray, n_steps: int, step_size: float,
                        rng: np.random.Generator, min_bins_per_block: int = 5) -> Tuple[np.ndarray, int, int]:
    """
    Inject n_steps steps (i.e., n_injected_blocks = n_steps + 1) into base_flux.
    Steps are placed at random start indices ensuring at least min_bins_per_block per block
    where possible. Returns (flux_injected, n_injected_blocks, median_bins_per_block).
    """
    n_bins = len(base_flux)
    if n_bins <= 0:
        return base_flux.copy(), 1, 0

    # target blocks
    n_blocks = n_steps + 1
    # if too few bins, reduce blocks
    n_blocks = min(n_blocks, n_bins)
    # choose block breakpoints by selecting (n_blocks - 1) cut positions while enforcing minimum spacing
    # naive approach: iterate attempts until valid spacing found
    max_tries = 1000
    for attempt in range(max_tries):
        # pick cut points among bin indices 1..n_bins-1
        if n_blocks == 1:
            cuts = []
        else:
            cuts = sorted(rng.choice(np.arange(1, n_bins), size=n_blocks - 1, replace=False).tolist())
        # compute block lengths
        starts = [0] + cuts
        ends = cuts + [n_bins]
        lengths = [ends[i] - starts[i] for i in range(len(starts))]
        if all(L >= min_bins_per_block for L in lengths):
            break
    else:
        # fallback: evenly spaced
        bins_per_block = max(1, n_bins // n_blocks)
        starts = [i * bins_per_block for i in range(n_blocks)]
        ends = starts[1:] + [n_bins]
        lengths = [ends[i] - starts[i] for i in range(len(starts))]

    # apply multiplicative steps
    flux = base_flux.copy().astype(float)
    current = 1.0
    # generate per-block multipliers: start at 1.0, each subsequent block multiplied by step_size
    block_multipliers = [current]
    for b in range(1, n_blocks):
        current *= step_size
        block_multipliers.append(current)
    # assign
    for i, (s, e) in enumerate(zip(starts, ends)):
        flux[s:e] = flux[s:e] * block_multipliers[i]
    # small local randomness to mimic measurement noise (do not change expected mean too much)
    flux *= rng.normal(1.0, 0.05, size=n_bins)
    flux = np.clip(flux, a_min=0.0, a_max=None)
    median_bins_per_block = int(np.median(lengths)) if lengths else 0
    return flux, n_blocks, median_bins_per_block


def run_confusion_experiment(base_flux: np.ndarray,
                             base_errors: Optional[np.ndarray],
                             n_trials: int,
                             step_sizes: List[float],
                             n_steps_list: List[int],
                             rng_seed: Optional[int],
                             min_bins_per_block: int,
                             p0: float,
                             detector: str = "zscore") -> Tuple[pd.DataFrame, Dict]:
    """
    For each combination of step_size and n_steps, run n_trials random injections,
    detect blocks, and record trial-level results. Returns trial DataFrame and summary dict.
    """
    rng = np.random.default_rng(rng_seed)
    records = []

    for n_steps in n_steps_list:
        for step_size in step_sizes:
            for trial in range(n_trials):
                flux_inj, true_n_blocks, bins_per_block = inject_steps_random(
                    base_flux, n_steps, step_size, rng, min_bins_per_block=min_bins_per_block
                )
                # derive errors for injected flux (scale original errors if present)
                if base_errors is not None and len(base_errors) == len(flux_inj):
                    # scale errors proportionally to flux change
                    # avoid zeros
                    base_err = np.asarray(base_errors, dtype=float)
                    scale = np.divide(flux_inj, np.maximum(base_flux, 1e-9), out=np.ones_like(flux_inj), where=base_flux!=0)
                    errors_inj = np.maximum(1e-6, base_err * scale)
                else:
                    errors_inj = np.maximum(1e-6, np.abs(flux_inj) * 0.1)

                if detector == "zscore":
                    detected = detect_blocks_zscore(flux_inj, errors_inj, p0=p0)
                elif detector == "astropy":
                    if astropy_bayesian_blocks is None:
                        raise RuntimeError("astropy detector requested but astropy not available")
                    detected = detect_blocks_astropy(flux_inj)
                else:
                    raise ValueError("detector must be 'zscore' or 'astropy'")

                records.append({
                    "trial": trial,
                    "n_steps": n_steps,
                    "step_size": float(step_size),
                    "true_n_blocks": int(true_n_blocks),
                    "detected_n_blocks": int(detected),
                    "bins_per_block": int(bins_per_block),
                    "mean_flux": float(np.mean(flux_inj)),
                    "std_flux": float(np.std(flux_inj)),
                })

    df = pd.DataFrame.from_records(records)
    # summary metrics:
    summary = {}

    # confusion matrix (counts)
    confusion = pd.crosstab(df["true_n_blocks"], df["detected_n_blocks"])
    summary["confusion_counts"] = confusion

    # normalized confusion by true row
    confusion_frac = confusion.div(confusion.sum(axis=1).replace(0, 1), axis=0)
    summary["confusion_fractions"] = confusion_frac

    # detection efficiency per (true_n_blocks, step_size) -> proportion detected correctly
    efficiency = {}
    grouped = df.groupby(["true_n_blocks", "step_size"])
    for (true_b, step), grp in grouped:
        n = len(grp)
        k = (grp["detected_n_blocks"] == true_b).sum()
        lo, hi = wilson_interval(k, n, confidence=0.95)
        efficiency.setdefault(true_b, {})[step] = {"n": n, "k": k, "eff": k / n if n else 0.0, "ci_lo": lo, "ci_hi": hi}

    summary["detection_efficiency"] = efficiency

    # detection vs bins_per_block (bucketed)
    df["bins_bucket"] = pd.cut(df["bins_per_block"], bins=[0, 1, 2, 4, 8, 16, 32, 1000], right=True)
    bins_eff = {}
    grouped_b = df.groupby(["true_n_blocks", "bins_bucket"])
    for (true_b, bucket), grp in grouped_b:
        n = len(grp)
        k = (grp["detected_n_blocks"] == true_b).sum()
        lo, hi = wilson_interval(k, n, confidence=0.95)
        bins_eff.setdefault(true_b, {})[str(bucket)] = {"n": n, "k": k, "eff": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi}

    summary["bins_efficiency"] = bins_eff

    return df, summary


# ---------- I/O and plotting ----------

def plot_confusion_matrix(confusion_frac: pd.DataFrame, outpath: Optional[Path] = None, title: str = "Confusion matrix (rows=true)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion_frac.fillna(0.0).values, origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(confusion_frac.shape[1]))
    ax.set_yticks(np.arange(confusion_frac.shape[0]))
    ax.set_xticklabels(confusion_frac.columns.astype(str))
    ax.set_yticklabels(confusion_frac.index.astype(str))
    ax.set_xlabel("Detected number of blocks")
    ax.set_ylabel("True number of blocks")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, label="Fraction")
    # annotate
    for i in range(confusion_frac.shape[0]):
        for j in range(confusion_frac.shape[1]):
            val = confusion_frac.iloc[i, j]
            if not np.isnan(val) and val > 0:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_efficiency_vs_bins(bins_eff: Dict[int, Dict[str, Dict]], outpath: Optional[Path] = None, title="Detection efficiency vs bins-per-block"):
    # bins_eff: {true_b: {bucket_str: {eff, ci_lo, ci_hi, n}}}
    fig, ax = plt.subplots(figsize=(8, 5))
    for true_b, d in sorted(bins_eff.items()):
        buckets = []
        effs = []
        ci_l = []
        ci_h = []
        ns = []
        for bucket, statsd in sorted(d.items(), key=lambda x: x[0]):
            buckets.append(bucket)
            effs.append(statsd["eff"] if not np.isnan(statsd["eff"]) else 0.0)
            ci_l.append(statsd["ci_lo"] if not np.isnan(statsd["eff"]) else 0.0)
            ci_h.append(statsd["ci_hi"] if not np.isnan(statsd["eff"]) else 0.0)
            ns.append(statsd["n"])
        x = np.arange(len(buckets)) + 0.0 + 0.15 * (true_b % 3)  # jitter
        ax.errorbar(x, effs, yerr=[np.array(effs) - np.array(ci_l), np.array(ci_h) - np.array(effs)],
                    fmt='o-', label=f"true {true_b} blocks (n≈{sum(ns)})")
        ax.set_xticks(np.arange(len(buckets)))
        ax.set_xticklabels(buckets, rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Bins-per-block bucket")
    ax.set_ylabel("Detection efficiency (with 95% Wilson CI)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_detected_hist(df: pd.DataFrame, outpath: Optional[Path] = None, title="Detected block counts distribution"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["detected_n_blocks"], bins=range(int(df["detected_n_blocks"].max()) + 2), align='left', rwidth=0.8, color='C0', alpha=0.8)
    ax.set_xlabel("Detected number of blocks")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze margin-of-error and confusion across injected vs detected block counts.")
    p.add_argument("--input", "-i", type=str, default=None, help="Path to wtlike file containing source light curve (optional). If omitted, use synthetic baseline.")
    p.add_argument("--source", "-s", type=str, default=None, help="Source key/name to extract from input file (required if --input given).")
    p.add_argument("--n_trials", "-n", type=int, default=2000, help="Trials per (n_steps, step_size) configuration")
    p.add_argument("--step_sizes", "-ss", type=float, nargs="+", default=[1.5, 2.0, 3.0], help="List of multiplicative step sizes to test")
    p.add_argument("--n_steps", "-ns", type=int, nargs="+", default=[1, 2], help="List of number of injected steps (S) to test; true blocks = S+1")
    p.add_argument("--min_bins_per_block", type=int, default=5, help="Minimum bins per block when randomizing step positions")
    p.add_argument("--p0", type=float, default=0.05, help="Significance threshold for z-score detector")
    p.add_argument("--detector", choices=["zscore", "astropy"], default="zscore", help="Detector to use")
    p.add_argument("--seed", type=int, default=1, help="Random seed")
    p.add_argument("--out_csv", type=str, default="block_confusion_trials.csv", help="CSV to save trial-level results")
    p.add_argument("--out_prefix", type=str, default="block_confusion", help="Prefix for saved plots")
    p.add_argument("--n_bins", type=int, default=100, help="If synthetic baseline used: number of time bins")
    p.add_argument("--synthetic", action="store_true", help="Skip input and use synthetic baseline flux (flat)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.input and not args.synthetic:
        if not args.source:
            raise SystemExit("When --input is provided you must also provide --source")
        loaded = _load_file(Path(args.input))
        lc = find_source_lightcurve(loaded, args.source)
        base_flux = np.asarray(lc.get('flux', []), dtype=float)
        base_errors = lc.get('errors', None)
        if base_errors is not None:
            base_errors = np.asarray(base_errors, dtype=float)
            if base_errors.shape != base_flux.shape:
                base_errors = None
        if len(base_flux) == 0:
            raise SystemExit("Extracted light curve is empty")
    else:
        # synthetic baseline: flat unit flux with small noise
        base_flux = np.ones(args.n_bins, dtype=float)
        rng = np.random.default_rng(args.seed or 1)
        base_flux = base_flux * rng.normal(1.0, 0.02, size=args.n_bins)  # small baseline jitter
        base_errors = np.maximum(1e-6, np.abs(base_flux) * 0.05)

    print(f"Running confusion experiment: n_trials={args.n_trials} per config; detectors={args.detector}")
    df_trials, summary = run_confusion_experiment(
        base_flux=base_flux,
        base_errors=base_errors,
        n_trials=args.n_trials,
        step_sizes=args.step_sizes,
        n_steps_list=args.n_steps,
        rng_seed=args.seed,
        min_bins_per_block=args.min_bins_per_block,
        p0=args.p0,
        detector=args.detector
    )

    # save trial-level CSV
    df_trials.to_csv(args.out_csv, index=False)
    print(f"Saved trial-level records to {args.out_csv}")

    # save confusion matrix plot
    confusion_frac = summary['confusion_fractions']
    plot_confusion_matrix(confusion_frac, outpath=Path(f"{args.out_prefix}_confusion_matrix.png"))
    print(f"Saved confusion matrix to {args.out_prefix}_confusion_matrix.png")

    # save detection vs bins-per-block plot
    plot_efficiency_vs_bins(summary['bins_efficiency'], outpath=Path(f"{args.out_prefix}_eff_vs_bins.png"))
    print(f"Saved efficiency vs bins-per-block to {args.out_prefix}_eff_vs_bins.png")

    # save histogram
    plot_detected_hist(df_trials, outpath=Path(f"{args.out_prefix}_detected_hist.png"))
    print(f"Saved detected histogram to {args.out_prefix}_detected_hist.png")

    # print a short summary (detection efficiency table)
    print("\nDetection efficiency summary (true_blocks -> step_size -> eff [95% CI]):")
    for true_b, bystep in sorted(summary['detection_efficiency'].items()):
        print(f"True blocks = {true_b}:")
        for step, statsd in sorted(bystep.items()):
            eff = statsd['eff']
            lo = statsd['ci_lo']
            hi = statsd['ci_hi']
            print(f"  step {step:.3f}: eff={eff:.3%} 95%CI=({lo:.3%}, {hi:.3%})  n={statsd['n']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
