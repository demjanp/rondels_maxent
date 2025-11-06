from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging
import math

from multiprocessing import Pool, cpu_count

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.features import rasterize

from .archeodata import CRS_FILENAME

LOG = logging.getLogger(__name__)

ENVIRO_DIR = "data/enviro"
PROJECTION_DIR = "data/projection"
_TIF_EXTENSIONS = {".tif", ".tiff"}
_DEFAULT_NODATA = -9999.0
_TRANSFORM_TOL = 1e-6


@dataclass(frozen=True)
class StageEnviroResult:
	environment_layers: List[Path]
	projection_layers: Optional[List[Path]] = None


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


def _determine_cellsize(
	paths: List[Path],
	target_crs: CRS,
	min_cell_size: Optional[float] = None,
) -> float:
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

	if min_cell_size is not None:
		if min_cell_size <= 0:
			raise ValueError("min_cell_size must be a positive value when provided.")

		try:
			unit_factor = target_crs.linear_units_factor  # meters per target unit
		except (AttributeError, ValueError):
			unit_factor = None

		if unit_factor and math.isfinite(unit_factor) and unit_factor > 0:
			min_in_target_units = min_cell_size / unit_factor
		elif target_crs.is_geographic:
			ellipsoid = getattr(target_crs, "ellipsoid", None)
			if ellipsoid is None and getattr(target_crs, "geodetic_crs", None):
				ellipsoid = target_crs.geodetic_crs.ellipsoid
			radius = None
			if ellipsoid is not None:
				radius = getattr(ellipsoid, "mean_radius", None) or getattr(ellipsoid, "semi_major_metre", None)
			if radius is None or radius <= 0:
				radius = 6_371_008.7714  # authalic radius of Earth in meters
			min_in_target_units = math.degrees(min_cell_size / radius)
		else:
			min_in_target_units = min_cell_size

		min_cell = max(min_cell, min_in_target_units)

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


def _resolve_nodata(value: Optional[float]) -> float:
	if value is None:
		return _DEFAULT_NODATA

	try:
		numeric = float(value)
	except (TypeError, ValueError):
		return _DEFAULT_NODATA

	if math.isnan(numeric) or not math.isfinite(numeric):
		return _DEFAULT_NODATA

	return numeric


def _prepare_mask(mask_path: Path, target_crs: CRS) -> tuple[List[dict], Tuple[float, float, float, float]]:
	mask_path = Path(mask_path)
	if not mask_path.exists():
		raise FileNotFoundError(f"Mask file not found: {mask_path}")

	try:
		import geopandas as gpd
		from shapely.geometry import mapping
	except Exception as exc:  # pragma: no cover - import/runtime environment
		raise RuntimeError("geopandas and shapely are required to use --mask") from exc

	gdf = gpd.read_file(mask_path)
	if gdf.empty:
		raise ValueError(f"No features found in mask dataset: {mask_path}")
	if gdf.crs is None:
		raise ValueError(f"Mask dataset has no CRS defined: {mask_path}")

	gdf = gdf.to_crs(target_crs.to_string())
	bounds_raw = tuple(float(v) for v in gdf.total_bounds)
	if len(bounds_raw) != 4:
		raise ValueError(f"Failed to compute bounds for mask dataset: {mask_path}")
	minx, miny, maxx, maxy = bounds_raw
	if not all(math.isfinite(v) for v in bounds_raw):
		raise ValueError(f"Mask dataset bounds contain non-finite values: {bounds_raw}")
	if minx >= maxx or miny >= maxy:
		raise ValueError(f"Mask dataset has invalid bounds after reprojection: {bounds_raw}")

	try:
		gdf = gdf.explode(index_parts=False)
	except Exception:
		pass

	try:
		from shapely.validation import make_valid as _make_valid
	except Exception:  # pragma: no cover - optional dependency
		_make_valid = None  # type: ignore

	geoms: List[dict] = []
	for geom in gdf.geometry:
		if geom is None or geom.is_empty:
			continue
		if not geom.is_valid:
			try:
				geom = _make_valid(geom) if _make_valid else geom.buffer(0)
			except Exception:
				pass
		if geom is None or geom.is_empty:
			continue
		geom_type = getattr(geom, "geom_type", "")
		if geom_type == "Polygon":
			geoms.append(mapping(geom))
		elif geom_type == "MultiPolygon":
			for part in getattr(geom, "geoms", []):
				if part and not part.is_empty:
					geoms.append(mapping(part))
		elif geom_type == "GeometryCollection":
			for part in getattr(geom, "geoms", []):
				sub_type = getattr(part, "geom_type", "")
				if sub_type == "Polygon":
					geoms.append(mapping(part))
				elif sub_type == "MultiPolygon":
					for sub_part in getattr(part, "geoms", []):
						if sub_part and not sub_part.is_empty:
							geoms.append(mapping(sub_part))

	if not geoms:
		raise ValueError(f"No polygon geometries found in mask dataset: {mask_path}")

	return geoms, (minx, miny, maxx, maxy)


def _stage_single_enviro(
	task: tuple[int, int, str, str, int, int, tuple[float, ...], str, bool, Optional[List[dict]]]
) -> tuple[int, Path]:
	(
		index,
		total,
		tif_path_str,
		dst_path_str,
		width,
		height,
		dst_transform_gdal,
		target_crs_wkt,
		is_categorical,
		mask_shapes,
	) = task

	tif_path = Path(tif_path_str)
	dst_path = Path(dst_path_str)
	dst_transform = Affine.from_gdal(*dst_transform_gdal)
	target_crs = CRS.from_wkt(target_crs_wkt)

	LOG.info("Staging environmental data %d/%d from %s.", index, total, tif_path)

	with rasterio.open(tif_path) as src:
		if src.count != 1:
			raise ValueError(f"Expected single-band raster, found {src.count} bands in {tif_path}")
		if src.crs is None:
			raise ValueError(f"Input raster missing CRS: {tif_path}")

		src_nodata = src.nodata
		dst_nodata = _resolve_nodata(src_nodata)
		dst_data = np.full((height, width), dst_nodata, dtype=np.float32)

		reproject(
			source=rasterio.band(src, 1),
			destination=dst_data,
			src_transform=src.transform,
			src_crs=src.crs,
			dst_transform=dst_transform,
			dst_crs=target_crs,
			resampling=Resampling.nearest if is_categorical else Resampling.bilinear,
			src_nodata=src_nodata,
			dst_nodata=dst_nodata,
		)

	# Apply polygon mask if provided: cells outside polygons -> NoData
	if mask_shapes:
		mask = rasterize(
			mask_shapes,
			out_shape=(height, width),
			transform=dst_transform,
			fill=0,
			default_value=1,
			dtype="uint8",
			all_touched=False,
		)
		dst_data[mask == 0] = dst_nodata

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

	return index, dst_path


def _stage_raster_set(
	geotiff_paths: List[Path],
	dst_dir: Path,
	width: int,
	height: int,
	dst_transform: Affine,
	target_crs: CRS,
	cellsize: float,
	xllcorner: float,
	yllcorner: float,
	categorical_lookup: dict[str, str],
	found_categorical: set[str],
	mask_shapes: Optional[List[dict]],
	allow_skip_existing: bool,
	dataset_label: str,
) -> List[Path]:
	dst_dir = Path(dst_dir)
	dst_dir.mkdir(parents=True, exist_ok=True)

	output_paths_map: dict[int, Path] = {}
	to_process: List[tuple[int, int, str, str, int, int, tuple[float, ...], str, bool, Optional[List[dict]]]] = []
	total = len(geotiff_paths)
	target_crs_wkt = target_crs.to_wkt()
	dst_transform_gdal = dst_transform.to_gdal()

	for index, tif_path in enumerate(geotiff_paths, start=1):
		dst_path = dst_dir / f"{tif_path.stem}.asc"
		layer_key = tif_path.stem.lower()
		is_categorical = layer_key in categorical_lookup
		if is_categorical:
			found_categorical.add(layer_key)

		if (
			allow_skip_existing
			and not is_categorical
			and mask_shapes is None
			and _validate_existing(dst_path, width, height, cellsize, xllcorner, yllcorner)
		):
			LOG.info("Skipping existing %s layer %s.", dataset_label, dst_path)
			output_paths_map[index] = dst_path
			continue

		to_process.append(
			(
				index,
				total,
				str(tif_path),
				str(dst_path),
				width,
				height,
				dst_transform_gdal,
				target_crs_wkt,
				is_categorical,
				mask_shapes,
			)
		)

	if to_process:
		available = cpu_count() or 1
		processes = min(len(to_process), available)

		if processes > 1:
			with Pool(processes=processes) as pool:
				for idx, path in pool.map(_stage_single_enviro, to_process):
					output_paths_map[idx] = path
		else:
			idx, path = _stage_single_enviro(to_process[0])
			output_paths_map[idx] = path

	return [output_paths_map[i] for i in range(1, total + 1)]


def stage_enviro(
	enviro_dir: Path,
	out_dir: Path,
	upper_left: Tuple[float, float],
	lower_right: Tuple[float, float],
	min_cell_size: Optional[float] = None,
	categorical_layers: Optional[Iterable[str]] = None,
	mask_path: Optional[Path] = None,
) -> StageEnviroResult:
	"""Reproject environmental rasters and export them in aligned ASCII grid format.

	Parameters
	----------
	enviro_dir : Path
		Directory containing source GeoTIFFs.
	out_dir : Path
		Workspace directory that will receive the ASCII outputs.
	upper_left, lower_right : tuple[float, float]
		AOI bounds (in target CRS coordinates).
	min_cell_size : float, optional
		Minimum cell size in meters; auto-calculated when omitted or <= 0.
	categorical_layers : Iterable[str], optional
		Names (without extension) of rasters to treat as categorical, resampled with
		nearest neighbour to preserve class values.
	mask_path : Path, optional
		Path to a GPKG containing polygon/multipolygon features to mask the
		output rasters. Cells outside polygons are written as NoData.

	Returns
	-------
	StageEnviroResult
		Paths to staged environmental layers and optional projection layers.
	"""
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
	categorical_lookup = {
		name.lower(): name for name in categorical_layers or []
	}
	found_categorical: set[str] = set()

	cellsize = _determine_cellsize(geotiff_paths, target_crs, min_cell_size)

	mask_shapes: Optional[List[dict]] = None
	env_upper_left = (float(upper_left[0]), float(upper_left[1]))
	env_lower_right = (float(lower_right[0]), float(lower_right[1]))

	if mask_path is not None:
		mask_shapes, mask_bounds = _prepare_mask(mask_path, target_crs)
		minx, miny, maxx, maxy = mask_bounds
		env_upper_left = (minx, maxy)
		env_lower_right = (maxx, miny)
		LOG.info(
			"Using mask bounds for environmental layers: UL=(%.6f, %.6f), LR=(%.6f, %.6f)",
			env_upper_left[0],
			env_upper_left[1],
			env_lower_right[0],
			env_lower_right[1],
		)

	env_width, env_height, env_transform, env_xllcorner, env_yllcorner = _calculate_grid(
		env_upper_left, env_lower_right, cellsize
	)

	environment_layers = _stage_raster_set(
		geotiff_paths,
		asc_dir,
		env_width,
		env_height,
		env_transform,
		target_crs,
		cellsize,
		env_xllcorner,
		env_yllcorner,
		categorical_lookup,
		found_categorical,
		mask_shapes,
		allow_skip_existing=mask_shapes is None,
		dataset_label="environmental",
	)

	projection_layers: Optional[List[Path]] = None
	if mask_shapes is not None:
		projection_dir = out_dir / PROJECTION_DIR
		proj_width, proj_height, proj_transform, proj_xllcorner, proj_yllcorner = _calculate_grid(
			(float(upper_left[0]), float(upper_left[1])),
			(float(lower_right[0]), float(lower_right[1])),
			cellsize,
		)
		projection_layers = _stage_raster_set(
			geotiff_paths,
			projection_dir,
			proj_width,
			proj_height,
			proj_transform,
			target_crs,
			cellsize,
			proj_xllcorner,
			proj_yllcorner,
			categorical_lookup,
			found_categorical,
			mask_shapes=None,
			allow_skip_existing=False,
			dataset_label="projection",
		)

	if categorical_lookup:
		missing = set(categorical_lookup).difference(found_categorical)
		if missing:
			names = ", ".join(categorical_lookup[name] for name in sorted(missing))
			LOG.warning("Categorical layers not found in %s: %s", enviro_dir, names)

	return StageEnviroResult(
		environment_layers=environment_layers,
		projection_layers=projection_layers,
	)
