"""Visualize environmental niche overlap using Schoener's D statistic."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook


class AsciiGridError(Exception):
    """Raised when an ASC grid cannot be parsed."""


@dataclass
class AsciiGrid:
    """Representation of an ESRI ASCII grid."""

    data: np.ndarray
    nodata: Optional[float]
    ncols: int
    nrows: int
    xllcorner: float
    yllcorner: float
    cellsize: float


@dataclass
class StatisticSummary:
    mean: float
    ci_low: float
    ci_high: float


@dataclass
class OverlapResult:
    variable: str
    schoeners_d: StatisticSummary
    avg_niche1: StatisticSummary
    avg_niche2: StatisticSummary
    avg_diff: StatisticSummary
    avg_diff_z: StatisticSummary


BBox = Tuple[float, float, float, float]
FLOAT_TOL = 1e-6
NUM_BINS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate overlap histograms for two MaxEnt niche models across all "
            "environmental layers in a directory."
        )
    )
    parser.add_argument(
        "--niche1",
        required=True,
        type=Path,
        help=(
            "Directory containing a 'maxent' subfolder with site_#.asc rasters for "
            "the first niche (legacy: direct path to one .asc)."
        ),
    )
    parser.add_argument(
        "--niche2",
        required=True,
        type=Path,
        help=(
            "Directory containing a 'maxent' subfolder with site_#.asc rasters for "
            "the second niche (legacy: direct path to one .asc)."
        ),
    )
    parser.add_argument(
        "--enviro",
        required=True,
        type=Path,
        help="Directory containing environmental layers in .asc format.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where overlap plots (PNG) will be written.",
    )
    parser.add_argument(
        "--categorical",
        nargs="*",
        default=[],
        help=(
            "List of environmental layer names (without .asc) to treat as categorical. "
            "Supports comma-separated values."
        ),
    )
    return parser.parse_args()


def parse_categorical_layers(values: Sequence[str]) -> Set[str]:
    result: Set[str] = set()
    for value in values:
        for part in value.split(","):
            name = part.strip().lower()
            if name:
                result.add(name)
    return result


def load_niche_ensemble(niche_path: Path) -> List[AsciiGrid]:
    asc_paths = _collect_niche_files(niche_path)
    return [read_ascii_grid(path) for path in asc_paths]


def _collect_niche_files(niche_path: Path) -> List[Path]:
    if niche_path.is_file():
        if niche_path.suffix.lower() != ".asc":
            raise SystemExit(f"Niche path must be an .asc file or directory: {niche_path}")
        return [niche_path]

    if not niche_path.exists():
        raise SystemExit(f"Niche path does not exist: {niche_path}")
    if not niche_path.is_dir():
        raise SystemExit(f"Niche path must be a directory: {niche_path}")

    candidate_dir = niche_path / "maxent"
    if candidate_dir.is_dir():
        search_dir = candidate_dir
    else:
        search_dir = niche_path
    asc_paths = sorted(
        (p for p in search_dir.iterdir() if p.suffix.lower() == ".asc"),
        key=_site_sort_key,
    )
    if not asc_paths:
        raise SystemExit(f"No .asc files found in {search_dir}")
    return asc_paths


def _site_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return int(match.group(1)), path.name.lower()
    return sys.maxsize, path.name.lower()


def read_ascii_grid(path: Path) -> AsciiGrid:
    """Read an ESRI ASCII grid into memory as floats."""
    header: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for _ in range(6):
                line = handle.readline()
                if not line:
                    raise AsciiGridError(f"Incomplete header in {path}")
                parts = line.strip().split()
                if len(parts) < 2:
                    raise AsciiGridError(f"Malformed header line in {path}: {line.strip()}")
                header[parts[0].lower()] = parts[1]

            try:
                ncols = int(header["ncols"])
                nrows = int(header["nrows"])
                cellsize = float(header["cellsize"])
            except KeyError as exc:
                raise AsciiGridError(f"Missing nrows/ncols/cellsize in {path}") from exc

            nodata_value = header.get("nodata_value")
            nodata = float(nodata_value) if nodata_value is not None else None

            xllcorner = _origin_from_header(header, "x", cellsize, path)
            yllcorner = _origin_from_header(header, "y", cellsize, path)

            data = np.loadtxt(handle, dtype=float, ndmin=1)
    except FileNotFoundError as exc:
        raise AsciiGridError(f"File not found: {path}") from exc
    except ValueError as exc:
        raise AsciiGridError(f"Unable to parse numeric data in {path}") from exc

    data = np.asarray(data, dtype=float)
    expected_size = nrows * ncols
    if data.size != expected_size:
        raise AsciiGridError(
            f"Unexpected cell count in {path} (expected {expected_size}, got {data.size})"
        )

    data = data.reshape((nrows, ncols))
    if nodata is not None:
        mask = np.isclose(data, nodata, equal_nan=True)
        data = data.astype(float, copy=False)
        data[mask] = np.nan
    else:
        data = data.astype(float, copy=False)

    return AsciiGrid(
        data=data,
        nodata=nodata,
        ncols=ncols,
        nrows=nrows,
        xllcorner=xllcorner,
        yllcorner=yllcorner,
        cellsize=cellsize,
    )


def _origin_from_header(header: dict[str, str], axis: str, cellsize: float, path: Path) -> float:
    corner_key = f"{axis}llcorner"
    center_key = f"{axis}llcenter"
    try:
        if corner_key in header:
            return float(header[corner_key])
        if center_key in header:
            center = float(header[center_key])
            return center - cellsize / 2.0
    except ValueError as exc:
        raise AsciiGridError(f"Invalid {axis} origin in {path}") from exc
    raise AsciiGridError(f"Missing {corner_key}/{center_key} in {path}")


def list_env_layers(env_dir: Path) -> List[Path]:
    if not env_dir.exists():
        raise SystemExit(f"Environment directory does not exist: {env_dir}")
    if not env_dir.is_dir():
        raise SystemExit(f"Environment path must be a directory: {env_dir}")
    env_layers = sorted(p for p in env_dir.iterdir() if p.suffix.lower() == ".asc")
    if not env_layers:
        raise SystemExit(f"No .asc files found in {env_dir}")
    return env_layers


def grid_bbox(grid: AsciiGrid) -> BBox:
    x_min = grid.xllcorner
    x_max = grid.xllcorner + grid.ncols * grid.cellsize
    y_min = grid.yllcorner
    y_max = grid.yllcorner + grid.nrows * grid.cellsize
    return x_min, x_max, y_min, y_max


def clip_grid_to_bbox(grid: AsciiGrid, bbox: BBox, require_full: bool = True) -> Optional[AsciiGrid]:
    x_min, x_max, y_min, y_max = bbox
    gx_min, gx_max, gy_min, gy_max = grid_bbox(grid)
    tol = grid.cellsize * FLOAT_TOL

    if require_full and (
        x_min < gx_min - tol or x_max > gx_max + tol or y_min < gy_min - tol or y_max > gy_max + tol
    ):
        return None

    overlap_x_min = max(x_min, gx_min)
    overlap_x_max = min(x_max, gx_max)
    overlap_y_min = max(y_min, gy_min)
    overlap_y_max = min(y_max, gy_max)

    if overlap_x_min >= overlap_x_max or overlap_y_min >= overlap_y_max:
        return None

    left = max(0, int(round((overlap_x_min - gx_min) / grid.cellsize)))
    right = max(0, int(round((gx_max - overlap_x_max) / grid.cellsize)))
    top = max(0, int(round((gy_max - overlap_y_max) / grid.cellsize)))
    bottom = max(0, int(round((overlap_y_min - gy_min) / grid.cellsize)))

    row_end = grid.nrows - bottom if bottom > 0 else grid.nrows
    col_end = grid.ncols - right if right > 0 else grid.ncols
    sliced = grid.data[top:row_end, left:col_end]
    if sliced.size == 0:
        return None

    return AsciiGrid(
        data=sliced,
        nodata=grid.nodata,
        ncols=sliced.shape[1],
        nrows=sliced.shape[0],
        xllcorner=grid.xllcorner + left * grid.cellsize,
        yllcorner=grid.yllcorner + bottom * grid.cellsize,
        cellsize=grid.cellsize,
    )


def align_niches(niche1: AsciiGrid, niche2: AsciiGrid) -> Tuple[AsciiGrid, AsciiGrid, BBox]:
    if not np.isclose(niche1.cellsize, niche2.cellsize, rtol=0, atol=niche1.cellsize * FLOAT_TOL):
        raise SystemExit("Niche rasters must use the same cellsize to compute overlap.")

    x1_min, x1_max, y1_min, y1_max = grid_bbox(niche1)
    x2_min, x2_max, y2_min, y2_max = grid_bbox(niche2)

    overlap_x_min = max(x1_min, x2_min)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_min = max(y1_min, y2_min)
    overlap_y_max = min(y1_max, y2_max)

    if overlap_x_min >= overlap_x_max or overlap_y_min >= overlap_y_max:
        raise SystemExit("Niche rasters do not overlap spatially.")

    bbox = (overlap_x_min, overlap_x_max, overlap_y_min, overlap_y_max)

    clipped1 = clip_grid_to_bbox(niche1, bbox)
    clipped2 = clip_grid_to_bbox(niche2, bbox)
    if clipped1 is None or clipped2 is None:
        raise SystemExit("Failed to clip niches to the overlapping area.")

    return clipped1, clipped2, bbox


def align_niche_ensembles(
    niche1_members: Sequence[AsciiGrid],
    niche2_members: Sequence[AsciiGrid],
) -> Tuple[List[AsciiGrid], List[AsciiGrid], BBox]:
    if not niche1_members or not niche2_members:
        raise SystemExit("Both niches must contain at least one raster.")
    if len(niche1_members) != len(niche2_members):
        raise SystemExit("Niche ensembles must have the same number of rasters.")

    all_members = list(niche1_members) + list(niche2_members)
    cellsize = all_members[0].cellsize
    for member in all_members[1:]:
        if not np.isclose(member.cellsize, cellsize, rtol=0, atol=cellsize * FLOAT_TOL):
            raise SystemExit("All niche rasters must share the same cellsize.")

    bbox = _intersection_bbox(all_members)
    if bbox is None:
        raise SystemExit("Niche rasters do not overlap spatially.")

    aligned1 = []
    aligned2 = []
    for member in niche1_members:
        clipped = clip_grid_to_bbox(member, bbox)
        if clipped is None:
            raise SystemExit("Failed to align niche 1 rasters to the common overlap.")
        aligned1.append(clipped)
    for member in niche2_members:
        clipped = clip_grid_to_bbox(member, bbox)
        if clipped is None:
            raise SystemExit("Failed to align niche 2 rasters to the common overlap.")
        aligned2.append(clipped)
    return aligned1, aligned2, bbox


def _intersection_bbox(grids: Sequence[AsciiGrid]) -> Optional[BBox]:
    if not grids:
        return None
    x_min, x_max, y_min, y_max = grid_bbox(grids[0])
    for grid in grids[1:]:
        gx_min, gx_max, gy_min, gy_max = grid_bbox(grid)
        x_min = max(x_min, gx_min)
        x_max = min(x_max, gx_max)
        y_min = max(y_min, gy_min)
        y_max = min(y_max, gy_max)
        if x_min >= x_max or y_min >= y_max:
            return None
    return x_min, x_max, y_min, y_max


def average_ascii_grids(grids: Sequence[AsciiGrid]) -> AsciiGrid:
    if not grids:
        raise ValueError("Cannot average an empty list of grids.")
    data_stack = np.stack([grid.data for grid in grids], axis=0)
    mean_data = np.nanmean(data_stack, axis=0)
    template = grids[0]
    return AsciiGrid(
        data=mean_data,
        nodata=template.nodata,
        ncols=template.ncols,
        nrows=template.nrows,
        xllcorner=template.xllcorner,
        yllcorner=template.yllcorner,
        cellsize=template.cellsize,
    )


def combined_valid_values(env: np.ndarray, niche1: np.ndarray, niche2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if env.shape != niche1.shape or env.shape != niche2.shape:
        raise ValueError("Raster shapes do not match.")
    mask = np.isnan(env) | np.isnan(niche1) | np.isnan(niche2)
    valid = ~mask
    env_values = env[valid]
    weights1 = niche1[valid]
    weights2 = niche2[valid]
    return env_values, weights1, weights2


def compute_histograms(
    env_values: np.ndarray,
    weights1: np.ndarray,
    weights2: np.ndarray,
    bins: int = NUM_BINS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_val = float(np.nanmin(env_values))
    max_val = float(np.nanmax(env_values))
    if min_val == max_val:
        # Avoid zero-width bins when the env layer is constant.
        min_val -= 0.5
        max_val += 0.5

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    counts1, _ = np.histogram(env_values, bins=bin_edges, weights=weights1)
    counts2, _ = np.histogram(env_values, bins=bin_edges, weights=weights2)
    return counts1.astype(float), counts2.astype(float), bin_edges


def compute_categorical_histograms(
    env_values: np.ndarray,
    weights1: np.ndarray,
    weights2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    categories, inverse = np.unique(env_values, return_inverse=True)
    categories = categories.astype(float)
    counts1 = np.zeros_like(categories, dtype=float)
    counts2 = np.zeros_like(categories, dtype=float)
    np.add.at(counts1, inverse, weights1)
    np.add.at(counts2, inverse, weights2)
    return categories, counts1, counts2


def normalize_counts(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts / total


def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    total_weight = np.nansum(weights)
    if total_weight <= 0:
        return float("nan")
    return float(np.nansum(values * weights) / total_weight)


def schoeners_d(dist1: np.ndarray, dist2: np.ndarray) -> float:
    if dist1.size != dist2.size:
        raise ValueError("Distribution lengths do not match for Schoener's D.")
    return 1.0 - 0.5 * np.abs(dist1 - dist2).sum()


def build_distributions(
    env_values: np.ndarray,
    weights1: np.ndarray,
    weights2: np.ndarray,
    is_categorical: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if is_categorical:
        categories, counts1, counts2 = compute_categorical_histograms(env_values, weights1, weights2)
        dist1 = normalize_counts(counts1)
        dist2 = normalize_counts(counts2)
        return dist1, dist2, categories, None
    counts1, counts2, bin_edges = compute_histograms(env_values, weights1, weights2, NUM_BINS)
    dist1 = normalize_counts(counts1)
    dist2 = normalize_counts(counts2)
    return dist1, dist2, None, bin_edges


def compute_overlap_stats(
    env_values: np.ndarray,
    weights1: np.ndarray,
    weights2: np.ndarray,
    is_categorical: bool,
) -> Tuple[float, float, float, float, float]:
    if env_values.size == 0:
        return (float("nan"),) * 5
    dist1, dist2, _, _ = build_distributions(env_values, weights1, weights2, is_categorical)
    d_value = schoeners_d(dist1, dist2)
    avg1 = weighted_average(env_values, weights1)
    avg2 = weighted_average(env_values, weights2)
    avg_diff = avg1 - avg2 if np.isfinite(avg1) and np.isfinite(avg2) else float("nan")
    std_env = float(np.nanstd(env_values, ddof=0))
    if not np.isfinite(std_env) or std_env == 0.0 or not np.isfinite(avg_diff):
        avg_diff_z = float("nan")
    else:
        avg_diff_z = avg_diff / std_env
    return d_value, avg1, avg2, avg_diff, avg_diff_z


def plot_overlap(
    bin_edges: np.ndarray,
    dist1: np.ndarray,
    dist2: np.ndarray,
    label1: str,
    label2: str,
    variable_name: str,
    output_path: Path,
    avg1_summary: StatisticSummary,
    avg2_summary: StatisticSummary,
    d_summary: StatisticSummary,
) -> None:
    title = f"{variable_name} overlap"
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    widths = np.diff(bin_edges)

    fig, ax = plt.subplots(figsize=(6, 3))
    (line1,) = ax.plot(centers, dist1, label=label1, linewidth=2)
    ax.fill_between(
        centers,
        dist1,
        color=line1.get_color(),
        alpha=0.3,
    )
    (line2,) = ax.plot(centers, dist2, label=label2, linewidth=2)
    ax.fill_between(
        centers,
        dist2,
        color=line2.get_color(),
        alpha=0.3,
    )
    _draw_ci_band(ax, avg1_summary, line1.get_color())
    _draw_ci_band(ax, avg2_summary, line2.get_color())
    if np.isfinite(avg1_summary.mean):
        ax.axvline(
            avg1_summary.mean,
            color=line1.get_color(),
            linestyle="--",
            linewidth=1.5,
            label=f"{label1} avg",
        )
    if np.isfinite(avg2_summary.mean):
        ax.axvline(
            avg2_summary.mean,
            color=line2.get_color(),
            linestyle="--",
            linewidth=1.5,
            label=f"{label2} avg",
        )

    ax.set_xlabel(variable_name)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    ax.text(
        0.98,
        0.95,
        _format_d_text(d_summary),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_categorical_overlap(
    categories: np.ndarray,
    dist1: np.ndarray,
    dist2: np.ndarray,
    label1: str,
    label2: str,
    variable_name: str,
    output_path: Path,
    avg1_summary: StatisticSummary,
    avg2_summary: StatisticSummary,
    d_summary: StatisticSummary,
    category_labels: Optional[Dict[float, str]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.7
    bars1 = ax.bar(
        categories - width / 2,
        dist1,
        width=width,
        label=label1,
        alpha=0.7,
        edgecolor="black",
    )
    bars2 = ax.bar(
        categories + width / 2,
        dist2,
        width=width,
        label=label2,
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_xlabel(variable_name)
    ax.set_ylabel("Probability")
    ax.set_title(f"{variable_name} overlap (categorical)")
    ax.set_xticks(categories)
    if category_labels:
        tick_labels = [
            category_labels.get(float(value), _format_category_label(value)) for value in categories
        ]
    else:
        tick_labels = [_format_category_label(value) for value in categories]
    ax.set_xticklabels(tick_labels, rotation=90)
    color1 = bars1.patches[0].get_facecolor() if bars1.patches else (0, 0, 0, 1)
    color2 = bars2.patches[0].get_facecolor() if bars2.patches else (0.5, 0.5, 0.5, 1)
    _draw_ci_band(ax, avg1_summary, color1)
    _draw_ci_band(ax, avg2_summary, color2)
    if np.isfinite(avg1_summary.mean):
        ax.axvline(
            avg1_summary.mean,
            color=color1,
            linestyle="--",
            linewidth=1.5,
            label=f"{label1} avg",
        )
    if np.isfinite(avg2_summary.mean):
        ax.axvline(
            avg2_summary.mean,
            color=color2,
            linestyle="--",
            linewidth=1.5,
            label=f"{label2} avg",
        )
    ax.legend()
    ax.text(
        0.98,
        0.95,
        _format_d_text(d_summary),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _draw_ci_band(ax, summary: StatisticSummary, color: Tuple[float, ...]) -> None:
    if summary is None:
        return
    lower = summary.ci_low
    upper = summary.ci_high
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
        ax.axvspan(lower, upper, color=color, alpha=0.15)


def _format_d_text(summary: StatisticSummary) -> str:
    if not np.isfinite(summary.mean):
        return "Schoener's D = N/A"
    return f"Schoener's D = {summary.mean:.3f}"


def summarize_statistics(
    values: Sequence[float],
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
) -> StatisticSummary:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        nan_value = float("nan")
        return StatisticSummary(nan_value, nan_value, nan_value)
    array = np.asarray(finite_values, dtype=float)
    mean = float(np.mean(array))
    if array.size == 1:
        return StatisticSummary(mean, mean, mean)
    ci_low = float(np.percentile(array, lower_percentile))
    ci_high = float(np.percentile(array, upper_percentile))
    return StatisticSummary(mean, ci_low, ci_high)


def process_layers(
    env_layers: Sequence[Path],
    niche1_members: Sequence[AsciiGrid],
    niche2_members: Sequence[AsciiGrid],
    mean_niche1: AsciiGrid,
    mean_niche2: AsciiGrid,
    output_dir: Path,
    bbox: BBox,
    niche1_name: str,
    niche2_name: str,
    categorical_layers: Set[str],
    ci_percentiles: Tuple[float, float] = (2.5, 97.5),
) -> List[OverlapResult]:
    generated = 0
    overlaps: List[OverlapResult] = []
    lower_pct, upper_pct = ci_percentiles
    niche1_stack = np.stack([member.data for member in niche1_members], axis=0)
    niche2_stack = np.stack([member.data for member in niche2_members], axis=0)

    for env_path in env_layers:
        try:
            env_grid = read_ascii_grid(env_path)
        except AsciiGridError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            continue

        clipped_env = clip_grid_to_bbox(env_grid, bbox, require_full=True)
        if clipped_env is None:
            print(
                f"[skip] {env_path} does not fully cover the niche overlap extent",
                file=sys.stderr,
            )
            continue

        plot_env_values, plot_weights1, plot_weights2 = combined_valid_values(
            clipped_env.data, mean_niche1.data, mean_niche2.data
        )
        if plot_env_values.size == 0:
            print(f"[warn] No overlapping valid cells in {env_path}", file=sys.stderr)
            continue

        layer_name = env_path.stem
        is_categorical = layer_name.lower() in categorical_layers
        category_labels = load_category_labels(env_path) if is_categorical else None
        dist1, dist2, categories, bin_edges = build_distributions(
            plot_env_values, plot_weights1, plot_weights2, is_categorical
        )

        d_samples: List[float] = []
        avg1_samples: List[float] = []
        avg2_samples: List[float] = []
        avg_diff_samples: List[float] = []
        avg_diff_z_samples: List[float] = []

        for idx in range(niche1_stack.shape[0]):
            env_values, weights1, weights2 = combined_valid_values(
                clipped_env.data, niche1_stack[idx], niche2_stack[idx]
            )
            if env_values.size == 0:
                continue
            d_value, avg1, avg2, avg_diff, avg_diff_z = compute_overlap_stats(
                env_values, weights1, weights2, is_categorical
            )
            d_samples.append(d_value)
            avg1_samples.append(avg1)
            avg2_samples.append(avg2)
            avg_diff_samples.append(avg_diff)
            avg_diff_z_samples.append(avg_diff_z)

        if not d_samples:
            print(
                f"[warn] No valid statistics computed for {env_path} across niche ensembles",
                file=sys.stderr,
            )
            continue

        d_summary = summarize_statistics(d_samples, lower_pct, upper_pct)
        avg1_summary = summarize_statistics(avg1_samples, lower_pct, upper_pct)
        avg2_summary = summarize_statistics(avg2_samples, lower_pct, upper_pct)
        avg_diff_summary = summarize_statistics(avg_diff_samples, lower_pct, upper_pct)
        avg_diff_z_summary = summarize_statistics(avg_diff_z_samples, lower_pct, upper_pct)

        output_path = output_dir / f"{env_path.stem}_overlap.png"
        if is_categorical:
            if categories is None:
                print(
                    f"[warn] Unable to plot categorical overlap for {env_path}",
                    file=sys.stderr,
                )
                continue
            plot_categorical_overlap(
                categories,
                dist1,
                dist2,
                f"{env_path.stem} - {niche1_name}",
                f"{env_path.stem} - {niche2_name}",
                str(env_path.stem),
                output_path,
                avg1_summary,
                avg2_summary,
                d_summary,
                category_labels,
            )
        else:
            if bin_edges is None:
                print(
                    f"[warn] Unable to plot continuous overlap for {env_path}",
                    file=sys.stderr,
                )
                continue
            plot_overlap(
                bin_edges,
                dist1,
                dist2,
                f"{env_path.stem} - {niche1_name}",
                f"{env_path.stem} - {niche2_name}",
                str(env_path.stem),
                output_path,
                avg1_summary,
                avg2_summary,
                d_summary,
            )
        generated += 1
        overlaps.append(
            OverlapResult(
                variable=env_path.stem,
                schoeners_d=d_summary,
                avg_niche1=avg1_summary,
                avg_niche2=avg2_summary,
                avg_diff=avg_diff_summary,
                avg_diff_z=avg_diff_z_summary,
            )
        )
        print(f"[ok] {output_path}")

    if generated == 0:
        raise SystemExit("No plots were generated. Check input rasters.")
    return overlaps


def write_ascii_grid(grid: AsciiGrid, destination: Path) -> None:
    nodata_value = grid.nodata if grid.nodata is not None else -9999.0
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as handle:
        handle.write(f"ncols         {grid.ncols}\n")
        handle.write(f"nrows         {grid.nrows}\n")
        handle.write(f"xllcorner     {grid.xllcorner:.10f}\n")
        handle.write(f"yllcorner     {grid.yllcorner:.10f}\n")
        handle.write(f"cellsize      {grid.cellsize:.10f}\n")
        handle.write(f"NODATA_value  {nodata_value}\n")

        data = np.array(grid.data, copy=True, dtype=float)
        mask = np.isnan(data)
        data[mask] = nodata_value
        np.savetxt(handle, data, fmt="%.6f")


def save_difference_map(
    niche1: AsciiGrid,
    niche2: AsciiGrid,
    output_dir: Path,
    niche1_name: str,
    niche2_name: str,
) -> None:
    difference = np.where(
        np.isnan(niche1.data) | np.isnan(niche2.data),
        np.nan,
        niche1.data - niche2.data,
    )
    diff_grid = AsciiGrid(
        data=difference,
        nodata=niche1.nodata if niche1.nodata is not None else niche2.nodata,
        ncols=niche1.ncols,
        nrows=niche1.nrows,
        xllcorner=niche1.xllcorner,
        yllcorner=niche1.yllcorner,
        cellsize=niche1.cellsize,
    )
    diff_name = output_dir / f"{niche1_name}_minus_{niche2_name}.asc"
    write_ascii_grid(diff_grid, diff_name)


def write_overlap_table(
    overlaps: Sequence[OverlapResult],
    destination: Path,
    niche1_name: str,
    niche2_name: str,
) -> None:
    if not overlaps:
        return
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Overlap"
    sheet.append(
        [
            "Variable",
            "Schoener's D (mean)",
            "Schoener's D (CI low)",
            "Schoener's D (CI high)",
            f"Avg {niche1_name} (mean)",
            f"Avg {niche1_name} (CI low)",
            f"Avg {niche1_name} (CI high)",
            f"Avg {niche2_name} (mean)",
            f"Avg {niche2_name} (CI low)",
            f"Avg {niche2_name} (CI high)",
            "AVG_diff (mean)",
            "AVG_diff (CI low)",
            "AVG_diff (CI high)",
            "AVG_diff_zscore (mean)",
            "AVG_diff_zscore (CI low)",
            "AVG_diff_zscore (CI high)",
        ]
    )
    for overlap in overlaps:
        sheet.append(
            [
                overlap.variable,
                *_summary_cells(overlap.schoeners_d),
                *_summary_cells(overlap.avg_niche1),
                *_summary_cells(overlap.avg_niche2),
                *_summary_cells(overlap.avg_diff),
                *_summary_cells(overlap.avg_diff_z),
            ]
        )
    workbook.save(destination)


def _format_number(value: float) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)


def _summary_cells(summary: StatisticSummary) -> List[Optional[float]]:
    return [
        _format_number(summary.mean),
        _format_number(summary.ci_low),
        _format_number(summary.ci_high),
    ]


def _format_category_label(value: float) -> str:
    value_float = float(value)
    if value_float.is_integer():
        return str(int(value_float))
    return f"{value_float:g}"


def load_category_labels(layer_path: Path) -> Optional[Dict[float, str]]:
    label_path = layer_path.with_suffix(".labels")
    if not label_path.exists():
        return None
    labels: Dict[float, str] = {}
    try:
        with label_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 2:
                    continue
                value_text = row[0].strip()
                label_text = row[1].strip()
                if not value_text:
                    continue
                try:
                    value = float(value_text)
                except ValueError:
                    continue
                labels[value] = label_text or value_text
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[warn] Failed to read labels for {layer_path}: {exc}", file=sys.stderr)
        return None
    return labels or None


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    categorical_layers = parse_categorical_layers(args.categorical)
    niche1_members = load_niche_ensemble(args.niche1)
    niche2_members = load_niche_ensemble(args.niche2)
    niche1_aligned, niche2_aligned, bbox = align_niche_ensembles(niche1_members, niche2_members)
    niche1_mean = average_ascii_grids(niche1_aligned)
    niche2_mean = average_ascii_grids(niche2_aligned)
    niche1_name = args.niche1.stem
    niche2_name = args.niche2.stem
    save_difference_map(niche1_mean, niche2_mean, args.output, niche1_name, niche2_name)

    env_layers = list_env_layers(args.enviro)
    overlaps = process_layers(
        env_layers,
        niche1_aligned,
        niche2_aligned,
        niche1_mean,
        niche2_mean,
        args.output,
        bbox,
        niche1_name,
        niche2_name,
        categorical_layers,
    )
    write_overlap_table(overlaps, args.output / "overlaps.xlsx", niche1_name, niche2_name)


if __name__ == "__main__":
    main()
