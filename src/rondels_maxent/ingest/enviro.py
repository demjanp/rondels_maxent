from __future__ import annotations

from typing import List, Tuple
from pathlib import Path
import logging
import math

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

from .archeodata import CRS_FILENAME

LOG = logging.getLogger(__name__)

ENVIRO_DIR = "data/enviro"
_TIF_EXTENSIONS = {".tif", ".tiff"}
_DEFAULT_NODATA = -9999.0
_TRANSFORM_TOL = 1e-6


def _load_target_crs(out_dir: Path) -> CRS:
	crs_path = Path(out_dir) / CRS_FILENAME
	if not crs_path.exists():
		raise FileNotFoundError(f"CRS file not found: {crs_path}")

	crs_text = crs_path.read_text(encoding="utf-8").strip()
	if not crs_text:
		raise ValueError(f"CRS file is empty: {crs_path}")

	try:
		return CRS.from_wkt(crs_text)
	except Exception as exc:  # pragma: no cover - rasterio parsing
		raise ValueError(f"Invalid CRS WKT in {crs_path}") from exc


def _collect_geotiffs(enviro_dir: Path) -> List[Path]:
	paths = [
		p for p in enviro_dir.iterdir()
		if p.is_file() and p.suffix.lower() in _TIF_EXTENSIONS
	]
	if not paths:
		raise ValueError(f"No GeoTIFF files found in {enviro_dir}")
	return sorted(paths)


def _determine_cellsize(paths: List[Path], target_crs: CRS) -> float:
	min_cell = math.inf
	for path in paths:
		with rasterio.open(path) as src:
			if src.crs is None:
				raise ValueError(f"Input raster missing CRS: {path}")
			dst_transform, _, _ = calculate_default_transform(
				src.crs, target_crs, src.width, src.height, *src.bounds
			)
			x_res = abs(dst_transform.a)
			y_res = abs(dst_transform.e)
			if x_res <= 0 or y_res <= 0:
				raise ValueError(f"Unable to determine raster resolution for {path}")
			cellsize = min(x_res, y_res)
			min_cell = min(min_cell, cellsize)

	if not math.isfinite(min_cell) or min_cell <= 0:
		raise ValueError("Failed to determine common output resolution for environmental layers.")

	return min_cell


def _calculate_grid(
	upper_left: Tuple[float, float],
	lower_right: Tuple[float, float],
	cellsize: float,
) -> tuple[int, int, Affine, float, float]:
	if len(upper_left) != 2 or len(lower_right) != 2:
		raise ValueError("upper_left and lower_right must each contain exactly two floats.")

	ul_x, ul_y = map(float, upper_left)
	lr_x, lr_y = map(float, lower_right)
	if lr_x <= ul_x or ul_y <= lr_y:
		raise ValueError("Invalid AOI bounds: ensure upper_left is north-west of lower_right.")

	x_span = lr_x - ul_x
	y_span = ul_y - lr_y
	if x_span <= 0 or y_span <= 0:
		raise ValueError("Invalid AOI span; check corner coordinates.")

	width = max(1, int(round(x_span / cellsize)))
	height = max(1, int(round(y_span / cellsize)))
	adjusted_lr_x = ul_x + width * cellsize
	adjusted_lr_y = ul_y - height * cellsize

	transform = from_origin(ul_x, ul_y, cellsize, cellsize)
	xllcorner = ul_x
	yllcorner = adjusted_lr_y

	if not math.isclose(adjusted_lr_x, lr_x, abs_tol=cellsize):
		LOG.warning("Adjusted AOI LR_X from %.6f to %.6f to align with cell size.", lr_x, adjusted_lr_x)
	if not math.isclose(adjusted_lr_y, lr_y, abs_tol=cellsize):
		LOG.warning("Adjusted AOI LR_Y from %.6f to %.6f to align with cell size.", lr_y, adjusted_lr_y)

	return width, height, transform, xllcorner, yllcorner


def _validate_existing(
	dst_path: Path,
	width: int,
	height: int,
	cellsize: float,
	xllcorner: float,
	yllcorner: float,
) -> bool:
	if not dst_path.exists():
		return False

	with rasterio.open(dst_path) as existing:
		transform = existing.transform
		match_grid = (
			existing.width == width
			and existing.height == height
			and math.isclose(transform.a, cellsize, abs_tol=_TRANSFORM_TOL)
			and math.isclose(transform.e, -cellsize, abs_tol=_TRANSFORM_TOL)
			and math.isclose(transform.c, xllcorner, abs_tol=_TRANSFORM_TOL)
		)
		if not match_grid:
			return False
		expected_f = yllcorner + height * cellsize
		if not math.isclose(transform.f, expected_f, abs_tol=_TRANSFORM_TOL):
			return False

	return True


def stage_enviro(
	enviro_dir: Path,
	out_dir: Path,
	upper_left: Tuple[float, float],
	lower_right: Tuple[float, float],
) -> List[Path]:
	"""Reproject environmental rasters and export them in aligned ASCII grid format."""
	enviro_dir = Path(enviro_dir)
	if not enviro_dir.exists():
		raise FileNotFoundError(f"Enviro directory not found: {enviro_dir}")
	if not enviro_dir.is_dir():
		raise NotADirectoryError(f"Enviro path is not a directory: {enviro_dir}")

	out_dir = Path(out_dir)
	asc_dir = out_dir / ENVIRO_DIR
	asc_dir.mkdir(parents=True, exist_ok=True)

	target_crs = _load_target_crs(out_dir)
	geotiff_paths = _collect_geotiffs(enviro_dir)
	cellsize = _determine_cellsize(geotiff_paths, target_crs)
	width, height, dst_transform, xllcorner, yllcorner = _calculate_grid(upper_left, lower_right, cellsize)

	output_paths: List[Path] = []
	total = len(geotiff_paths)
	for index, tif_path in enumerate(geotiff_paths, start=1):
		dst_path = asc_dir / f"{tif_path.stem}.asc"
		if _validate_existing(dst_path, width, height, cellsize, xllcorner, yllcorner):
			LOG.info("Skipping existing environmental layer %s.", dst_path)
			output_paths.append(dst_path)
			continue

		LOG.info("Staging environmental data %d/%d from %s.", index, total, tif_path)
		with rasterio.open(tif_path) as src:
			if src.count != 1:
				raise ValueError(f"Expected single-band raster, found {src.count} bands in {tif_path}")
			if src.crs is None:
				raise ValueError(f"Input raster missing CRS: {tif_path}")

			src_nodata = src.nodata
			dst_nodata = float(src_nodata) if src_nodata is not None else _DEFAULT_NODATA
			dst_data = np.full((height, width), dst_nodata, dtype=np.float32)

			reproject(
				source=rasterio.band(src, 1),
				destination=dst_data,
				src_transform=src.transform,
				src_crs=src.crs,
				dst_transform=dst_transform,
				dst_crs=target_crs,
				resampling=Resampling.bilinear,
				src_nodata=src_nodata,
				dst_nodata=dst_nodata,
			)

			with rasterio.open(
				dst_path,
				"w",
				driver="AAIGrid",
				width=width,
				height=height,
				count=1,
				dtype=dst_data.dtype,
				crs=target_crs,
				transform=dst_transform,
				nodata=dst_nodata,
			) as dst:
				dst.write(dst_data, 1)

		output_paths.append(dst_path)

	return output_paths
