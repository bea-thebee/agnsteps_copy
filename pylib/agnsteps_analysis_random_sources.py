"""
Random Sources Batch Analysis for AGN Steps

Analyze many random sources from the agnsteps pickle database and compare
Bayesian block structure, light curve metrics, and flux ratios across the sample.
"""

import os
import pickle
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('dark_background')
sns.set_theme('talk', font_scale=1.0)


@dataclass
class SourceBatchResult:
    source_name: str
    nbb: int
    classification: str
    mean_flux: float
    std_flux: float
    min_flux: float
    max_flux: float
    time_span_days: float
    flux_ratio_type: str
    flux_ratio: Optional[float]
    log_flux_ratio: Optional[float]
    overall_ratio: Optional[float]
    bump_ratio: Optional[float]
    num_transitions: int
    width_before_weeks: Optional[float]
    width_after_weeks: Optional[float]
    width_bump_weeks: Optional[float]
    margin_pass: Optional[bool]
    source_metadata: Dict[str, Any]


class MultiRandomSourceAnalyzer:
    """Analyze many random AGN sources and compare light-curve statistics."""

    def __init__(
        self,
        info_file: str = 'files/source_info_v1.pkl',
        n_sources: int = 500,
        seed: int = 42,
        output_dir: str = 'random_sources_analysis',
        margin_weeks: int = 50,
    ) -> None:
        self.info_file = Path(info_file)
        self.n_sources = n_sources
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.margin_weeks = margin_weeks

        self.db: Dict[str, Any] = {}
        self.selected_sources: List[str] = []
        self.results: List[SourceBatchResult] = []
        self.summary_df: Optional[pd.DataFrame] = None

    def load_database(self) -> bool:
        if not self.info_file.exists():
            print(f"Error: info file not found: {self.info_file}")
            return False

        with open(self.info_file, 'rb') as handle:
            self.db = pickle.load(handle)

        print(f"Loaded {len(self.db)} source entries from {self.info_file}")
        return True

    @staticmethod
    def _normalize_light_curve(lc: Any) -> pd.DataFrame:
        if isinstance(lc, dict):
            df = pd.DataFrame.from_dict(lc, orient='index')
        else:
            df = pd.DataFrame(lc).copy()

        if 'flux' not in df.columns and 'flux' in df.index:
            df = df.T

        return df

    @staticmethod
    def _extract_flux_values(df: pd.DataFrame) -> np.ndarray:
        values = df['flux'].values
        flux_array = np.array([
            fv[0] if hasattr(fv, '__len__') and not isinstance(fv, (str, bytes)) else fv
            for fv in values
        ], dtype=float)
        return flux_array

    @staticmethod
    def _extract_errors(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        errors = df['errors'].values
        lower = np.array([abs(e[0]) if isinstance(e, (list, tuple, np.ndarray)) else abs(e) for e in errors], dtype=float)
        upper = np.array([abs(e[1]) if isinstance(e, (list, tuple, np.ndarray)) else abs(e) for e in errors], dtype=float)
        return lower, upper

    def _analyze_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        flux_array = self._extract_flux_values(df)
        errors_lower, errors_upper = self._extract_errors(df)

        times = df['t'].values
        widths = df['tw'].values
        time_span = (times[-1] - times[0]) + widths[-1]

        return {
            'num_blocks': len(df),
            'flux_values': flux_array,
            'errors_lower': errors_lower,
            'errors_upper': errors_upper,
            'times': times,
            'widths': widths,
            'mean_flux': float(np.mean(flux_array)) if flux_array.size else np.nan,
            'std_flux': float(np.std(flux_array)) if flux_array.size else np.nan,
            'min_flux': float(np.min(flux_array)) if flux_array.size else np.nan,
            'max_flux': float(np.max(flux_array)) if flux_array.size else np.nan,
            'time_span_days': float(time_span),
        }

    def _classify_block_structure(self, num_blocks: int) -> str:
        if num_blocks == 1:
            return 'single_block'
        if num_blocks == 2:
            return 'single_step'
        if num_blocks == 3:
            return 'two_steps'
        return 'multi_step'

    def _calculate_flux_ratios(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        flux_values = structure['flux_values']
        widths_weeks = structure['widths'] / 7.0
        num_blocks = structure['num_blocks']
        ratios: Dict[str, Any] = {
            'flux_ratio_type': self._classify_block_structure(num_blocks),
            'flux_ratio': None,
            'log_flux_ratio': None,
            'overall_ratio': None,
            'bump_ratio': None,
            'num_transitions': max(0, num_blocks - 1),
            'width_before_weeks': None,
            'width_after_weeks': None,
            'width_bump_weeks': None,
            'margin_pass': None,
        }

        if num_blocks == 1:
            return ratios

        if num_blocks == 2:
            a, b = flux_values[0], flux_values[1]
            w1, w2 = widths_weeks[0], widths_weeks[1]
            ratio = b / a if a != 0 else np.inf
            ratios.update({
                'flux_ratio': float(ratio),
                'log_flux_ratio': float(np.log10(ratio)) if ratio > 0 else np.nan,
                'width_before_weeks': float(w1),
                'width_after_weeks': float(w2),
                'margin_pass': bool((w1 >= self.margin_weeks) and (w2 >= self.margin_weeks)),
            })
            return ratios

        if num_blocks == 3:
            a, b, c = flux_values[0], flux_values[1], flux_values[2]
            w1, w2, w3 = widths_weeks[0], widths_weeks[1], widths_weeks[2]
            overall_ratio = c / a if a != 0 else np.inf
            bump_ratio = 2 * b / (a + c) if (a + c) != 0 else np.inf
            ratios.update({
                'overall_ratio': float(overall_ratio),
                'bump_ratio': float(bump_ratio),
                'log_flux_ratio': float(np.log10(overall_ratio)) if overall_ratio > 0 else np.nan,
                'width_before_weeks': float(w1),
                'width_bump_weeks': float(w2),
                'width_after_weeks': float(w3),
                'margin_pass': bool((w1 >= self.margin_weeks) and (w3 >= self.margin_weeks)),
            })
            return ratios

        consecutive_ratios = []
        for i in range(num_blocks - 1):
            denom = flux_values[i]
            if denom != 0:
                consecutive_ratios.append(float(flux_values[i + 1] / denom))
        ratios['overall_ratio'] = float(flux_values[-1] / flux_values[0]) if flux_values[0] != 0 else np.inf
        ratios['flux_ratio'] = consecutive_ratios[0] if consecutive_ratios else None
        ratios['log_flux_ratio'] = float(np.log10(consecutive_ratios[0])) if consecutive_ratios and consecutive_ratios[0] > 0 else np.nan
        ratios['margin_pass'] = None
        return ratios

    def _source_has_valid_light_curve(self, entry: Dict[str, Any]) -> bool:
        lc = entry.get('light_curve')
        if lc is None:
            return False
        try:
            df = self._normalize_light_curve(lc)
            return len(df) > 1 and {'t', 'tw', 'flux', 'errors'}.issubset(df.columns)
        except Exception:
            return False

    def select_sources(self) -> None:
        random.seed(self.seed)
        valid_sources = [name for name, entry in self.db.items() if self._source_has_valid_light_curve(entry)]
        if not valid_sources:
            raise RuntimeError('No valid sources with light curves available in the database')

        if len(valid_sources) >= self.n_sources:
            self.selected_sources = random.sample(valid_sources, self.n_sources)
        else:
            self.selected_sources = random.choices(valid_sources, k=self.n_sources)

        print(f"Selected {len(self.selected_sources)} random sources for analysis")

    def analyze_source(self, source_name: str) -> Optional[SourceBatchResult]:
        entry = self.db.get(source_name)
        if entry is None:
            return None

        lc = self._normalize_light_curve(entry['light_curve'])
        structure = self._analyze_structure(lc)
        ratios = self._calculate_flux_ratios(structure)
        classification = self._classify_block_structure(structure['num_blocks'])

        return SourceBatchResult(
            source_name=source_name,
            nbb=structure['num_blocks'],
            classification=classification,
            mean_flux=structure['mean_flux'],
            std_flux=structure['std_flux'],
            min_flux=structure['min_flux'],
            max_flux=structure['max_flux'],
            time_span_days=structure['time_span_days'],
            flux_ratio_type=ratios['flux_ratio_type'],
            flux_ratio=ratios['flux_ratio'],
            log_flux_ratio=ratios['log_flux_ratio'],
            overall_ratio=ratios['overall_ratio'],
            bump_ratio=ratios['bump_ratio'],
            num_transitions=ratios['num_transitions'],
            width_before_weeks=ratios['width_before_weeks'],
            width_after_weeks=ratios['width_after_weeks'],
            width_bump_weeks=ratios['width_bump_weeks'],
            margin_pass=ratios['margin_pass'],
            source_metadata={
                'association': entry.get('association'),
                'ts': entry.get('ts'),
                'bbvar': entry.get('bbvar'),
                'eflux100': entry.get('eflux100'),
                'r95': entry.get('r95'),
            }
        )

    def run(self) -> pd.DataFrame:
        if not self.load_database():
            raise RuntimeError('Unable to load source database')
        self.select_sources()

        results: List[SourceBatchResult] = []
        for idx, source_name in enumerate(self.selected_sources, start=1):
            entry = self.db[source_name]
            if not self._source_has_valid_light_curve(entry):
                continue
            result = self.analyze_source(source_name)
            if result is not None:
                results.append(result)
            if idx % 50 == 0:
                print(f"  Processed {idx}/{len(self.selected_sources)} sources")

        self.results = results
        rows = []
        for result in results:
            rows.append({
                'source_name': result.source_name,
                'nbb': result.nbb,
                'classification': result.classification,
                'mean_flux': result.mean_flux,
                'std_flux': result.std_flux,
                'min_flux': result.min_flux,
                'max_flux': result.max_flux,
                'time_span_days': result.time_span_days,
                'flux_ratio_type': result.flux_ratio_type,
                'flux_ratio': result.flux_ratio,
                'log_flux_ratio': result.log_flux_ratio,
                'overall_ratio': result.overall_ratio,
                'bump_ratio': result.bump_ratio,
                'num_transitions': result.num_transitions,
                'width_before_weeks': result.width_before_weeks,
                'width_after_weeks': result.width_after_weeks,
                'width_bump_weeks': result.width_bump_weeks,
                'margin_pass': result.margin_pass,
                'association': result.source_metadata.get('association'),
                'ts': result.source_metadata.get('ts'),
                'bbvar': result.source_metadata.get('bbvar'),
                'eflux100': result.source_metadata.get('eflux100'),
                'r95': result.source_metadata.get('r95'),
            })

        df = pd.DataFrame(rows)
        self.summary_df = df
        return df

    def save_summary(self, filename: str = 'random_sources_summary.csv') -> Path:
        if self.summary_df is None:
            raise RuntimeError('No summary DataFrame available; run analysis first')
        output_path = self.output_dir / filename
        self.summary_df.to_csv(output_path, index=False)
        print(f"Saved summary CSV to {output_path}")
        return output_path

    def plot_summary(self) -> None:
        if self.summary_df is None:
            raise RuntimeError('No summary DataFrame available; run analysis first')

        df = self.summary_df
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        sns.histplot(df['nbb'], bins=range(1, int(df['nbb'].max()) + 2), ax=axes[0, 0], color='#4c72b0')
        axes[0, 0].set_title('Distribution of Bayesian Block Count')
        axes[0, 0].set_xlabel('Number of Blocks')

        sns.histplot(df.loc[df['flux_ratio_type'] == 'single_step', 'flux_ratio'], bins=30, ax=axes[0, 1], color='#55a868')
        axes[0, 1].set_title('Flux Ratio Distribution for Single-Step Sources')
        axes[0, 1].set_xlabel('After/Before Flux Ratio')

        sns.scatterplot(data=df, x='mean_flux', y='std_flux', hue='classification', palette='tab10', ax=axes[1, 0])
        axes[1, 0].set_title('Mean vs. Std Flux by Classification')
        axes[1, 0].set_xlabel('Mean Flux')
        axes[1, 0].set_ylabel('Std Flux')
        axes[1, 0].legend(loc='best')

        sns.histplot(df['overall_ratio'].dropna(), bins=40, ax=axes[1, 1], color='#dd8452')
        axes[1, 1].set_title('Overall Flux Ratio Distribution (Multi-Step)')
        axes[1, 1].set_xlabel('Overall Flux Ratio')

        plt.tight_layout()
        outpath = self.output_dir / 'random_sources_summary.png'
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved summary plot to {outpath}")

    def plot_examples(self, n_examples: int = 9, filename: str = 'random_sources_examples.png') -> Path:
        if self.summary_df is None:
            raise RuntimeError('No summary DataFrame available; run analysis first')

        examples = self.summary_df.sort_values('nbb').head(n_examples)
        n = len(examples)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5), squeeze=False)
        axes = axes.flatten()

        for ax in axes[n:]:
            ax.axis('off')

        for idx, (_, row) in enumerate(examples.iterrows()):
            source_name = row['source_name']
            lc = self._normalize_light_curve(self.db[source_name]['light_curve'])
            flux_values = self._extract_flux_values(lc)
            times = lc['t'].values
            widths = lc['tw'].values
            ax = axes[idx]

            ax.errorbar(times, flux_values,
                        xerr=widths / 2,
                        fmt='o-', markersize=4,
                        color='cyan', alpha=0.8)
            ax.set_title(f"{source_name}\nnbb={row['nbb']} type={row['classification']}", fontsize=10)
            ax.set_xlabel('MJD')
            ax.set_ylabel('Flux')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        outpath = self.output_dir / filename
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved example light curve plot to {outpath}")
        return outpath


if __name__ == '__main__':
    analyzer = MultiRandomSourceAnalyzer(n_sources=500, seed=42, output_dir='random_sources_analysis')
    df = analyzer.run()
    analyzer.save_summary()
    analyzer.plot_summary()
    analyzer.plot_examples(n_examples=9)
    print('\nSummary statistics:')
    print(df.describe(include='all'))
