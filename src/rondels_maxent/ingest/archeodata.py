from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, cast

import geopandas as gpd
import pandas as pd

LOG = logging.getLogger(__name__)

COORDS: Mapping[str, str] = {"x": "WGS84_E", "y": "WGS84_N"}
SAMPLES_FILENAME = "data/samples.csv"
CRS_FILENAME = "data/crs.prj"
RULES_FILENAME = "rules.json"

def _load_rules(rules_path: Path) -> dict[str, list[str]]:
	rules_path = Path(rules_path)
	if not rules_path.exists():
		raise FileNotFoundError(f"Input rules not found: {rules_path}")
	if not rules_path.is_file():
		raise ValueError(f"Input rules path is not a file: {rules_path}")

	try:
		rules_data = json.loads(rules_path.read_text(encoding="utf-8"))
	except json.JSONDecodeError as exc:
		raise ValueError(f"Rules file is not valid JSON: {rules_path}") from exc

	if not isinstance(rules_data, dict):
		raise ValueError("Rules JSON must be an object mapping fields to allowed values.")

	validated_rules: dict[str, list[str]] = {}
	for field, values in rules_data.items():
		if not isinstance(field, str):
			raise ValueError("Rules JSON contains non-string field names.")
		if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
			raise ValueError(f"Rules for '{field}' must be a list of strings.")
		if not values:
			raise ValueError(f"Rules for '{field}' may not be empty.")
		validated_rules[field] = values

	return validated_rules


def stage_archeodata(in_path: Path, rules_path: Path, out_dir: Path):
	"""Stage archeological data for downstream processing."""
	in_path = Path(in_path)
	if not in_path.exists():
		raise FileNotFoundError(f"Input data not found: {in_path}")
	if not in_path.is_file():
		raise ValueError(f"Input path is not a file: {in_path}")

	rules = _load_rules(rules_path)

	LOG.info("Staging archeological data from %s.", in_path)
	
	try:
		gdf = gpd.read_file(in_path)
	except Exception as exc:  # pragma: no cover - geopandas internal errors
		raise RuntimeError(f"Failed to read geopackage: {in_path}") from exc

	required_columns = set(rules.keys()) | set(COORDS.values())
	missing_columns = [column for column in required_columns if column not in gdf.columns]
	if missing_columns:
		raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

	filtered = gdf.copy()
	for field, allowed_values in rules.items():
		filtered = filtered[filtered[field].isin(allowed_values)]

	if filtered.empty:
		raise ValueError("No records match the input rules.")

	x_column = COORDS.get("x")
	y_column = COORDS.get("y")
	if x_column is None or y_column is None:
		raise ValueError("Coordinate field mapping is incomplete.")

	if filtered[x_column].isna().any() or filtered[y_column].isna().any():
		raise ValueError("Coordinate columns contain missing values.")

	try:
		x_series = cast(pd.Series, pd.to_numeric(filtered[x_column], errors="raise"))
		y_series = cast(pd.Series, pd.to_numeric(filtered[y_column], errors="raise"))
		x_values = x_series.astype(float)
		y_values = y_series.astype(float)
	except (TypeError, ValueError) as exc:
		raise ValueError("Coordinate columns must contain numeric values.") from exc

	filtered = filtered.reset_index(drop=True)
	samples_df = pd.DataFrame({"site": ["site"] * len(filtered), "x": x_values, "y": y_values})

	data_dir = Path(out_dir) / "data"
	data_dir.mkdir(parents=True, exist_ok=True)

	samples_path = Path(out_dir) / SAMPLES_FILENAME
	samples_df.to_csv(samples_path, index=False)

	geometry = gpd.points_from_xy(samples_df["x"], samples_df["y"])
	filtered_gdf = gpd.GeoDataFrame(
		filtered.drop(columns="geometry", errors="ignore"),
		geometry=geometry,
		crs=gdf.crs,
	)

	gpkg_path = Path(out_dir) / f"{out_dir.name}.gpkg"
	filtered_gdf.to_file(gpkg_path, driver="GPKG", layer=out_dir.name)
	
	crs = gdf.crs
	if crs is None:
		raise ValueError("Input dataset lacks a defined CRS.")

	crs_path = Path(out_dir) / CRS_FILENAME
	crs_path.write_text(crs.to_wkt(version="WKT1_ESRI"), encoding="utf-8")
	
	output_rules_path = Path(out_dir) / RULES_FILENAME
	output_rules_path.write_text(json.dumps(rules, indent=2, sort_keys=True), encoding="utf-8")
	
	
