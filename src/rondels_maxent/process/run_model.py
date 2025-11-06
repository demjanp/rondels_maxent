from __future__ import annotations

from typing import List, Optional
from pathlib import Path
import csv
import shutil
import logging
import subprocess

from ..ingest.archeodata import SAMPLES_FILENAME, CRS_FILENAME

LOG = logging.getLogger(__name__)

MAX_MEMORY = "10g"
MAXENT_OUT_DIR = "maxent"

_MAXENT_JAR = "maxent.jar"
_EXPECTED_SAMPLE_HEADER = ("site", "x", "y")
_ASCII_SUFFIX = ".asc"


def _validate_env_layers(enviro_layers: List[Path]) -> List[Path]:
	if not enviro_layers:
		raise ValueError("No environmental layers were provided.")

	resolved_layers = []
	for layer in enviro_layers:
		layer_path = Path(layer)
		if not layer_path.exists():
			raise FileNotFoundError(f"Environmental layer not found: {layer_path}")
		if not layer_path.is_file():
			raise ValueError(f"Environmental layer is not a file: {layer_path}")
		if layer_path.suffix.lower() != _ASCII_SUFFIX:
			raise ValueError(f"Environmental layer must be an ASCII grid (.asc): {layer_path}")
		resolved_layers.append(layer_path)
	return resolved_layers


def _determine_base_dir(layer_path: Path) -> Path:
	try:
		return layer_path.parents[2]
	except IndexError as exc:
		raise ValueError(f"Environmental layer path has unexpected structure: {layer_path}") from exc


def _validate_samples_file(base_dir: Path) -> Path:
	samples_path = base_dir / SAMPLES_FILENAME
	if not samples_path.exists():
		raise FileNotFoundError(f"Samples file not found: {samples_path}")
	if not samples_path.is_file():
		raise ValueError(f"Samples path is not a file: {samples_path}")

	with samples_path.open("r", encoding="utf-8") as handle:
		reader = csv.reader(handle)
		try:
			header = next(reader)
		except StopIteration as exc:
			raise ValueError(f"Samples file is empty: {samples_path}") from exc
		if tuple(col.strip() for col in header) != _EXPECTED_SAMPLE_HEADER:
			raise ValueError(
				f"Samples file header must be '{','.join(_EXPECTED_SAMPLE_HEADER)}', "
				f"found: {header}"
			)

	return samples_path


def _ensure_single_parent(layers: List[Path]) -> Path:
	parent_dirs = {layer.parent for layer in layers}
	if len(parent_dirs) != 1:
		raise ValueError("All environmental layers must reside in the same directory.")
	return parent_dirs.pop()


def _prepare_maxent_paths(maxent_dir: Path) -> Path:
	maxent_dir = Path(maxent_dir)
	if not maxent_dir.exists():
		raise FileNotFoundError(f"MaxEnt directory not found: {maxent_dir}")
	if not maxent_dir.is_dir():
		raise NotADirectoryError(f"MaxEnt path is not a directory: {maxent_dir}")

	jar_path = maxent_dir / _MAXENT_JAR
	if not jar_path.exists():
		raise FileNotFoundError(f"MaxEnt jar not found: {jar_path}")
	if not jar_path.is_file():
		raise ValueError(f"MaxEnt jar path is not a file: {jar_path}")

	return jar_path


def run_model(
	enviro_layers: List[Path],
	maxent_dir: Path,
	out_dir: Path,
	categorical: Optional[List[str]] = None,
	projection_layers: Optional[List[Path]] = None,
) -> None:
	"""Execute MaxEnt model using prepared samples and environmental layers.

	Parameters
	----------
	enviro_layers : list[Path]
		Paths to environmental layer files (.asc, .tif, etc.)
	maxent_dir : Path
		Directory containing maxent.jar.
	out_dir : Path
		Base output directory.
	categorical : list[str], optional
		Names (without extension) of layers to treat as categorical.
		Example: ['HWSD2', 'landcover']
	projection_layers : list[Path], optional
		Paths to projection rasters that will be supplied to MaxEnt via the
		`projectionlayers` parameter when provided.
	"""
	resolved_layers = _validate_env_layers(enviro_layers)
	layer_dir = _ensure_single_parent(resolved_layers)
	projection_dir: Optional[Path] = None
	if projection_layers:
		resolved_projection_layers = _validate_env_layers(projection_layers)
		projection_dir = _ensure_single_parent(resolved_projection_layers).resolve()
	base_dir = Path(out_dir)
	samples_path = _validate_samples_file(base_dir)
	maxent_jar = _prepare_maxent_paths(maxent_dir)

	output_dir = (base_dir / MAXENT_OUT_DIR).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	command = [
		"java",
		f"-mx{MAX_MEMORY}",
		"-jar",
		str(maxent_jar),
		f"environmentallayers={str(layer_dir.resolve())}",
		f"samplesfile={str(samples_path.resolve())}",
		f"outputdirectory={str(output_dir)}",
		"jackknife=true",
		"responsecurves=true",
		"pictures=true",
		"visible=false",
		"plots=true",
		"writebackgroundpredictions=true",
		"askoverwrite=false",
		"autorun",
	]

	if projection_dir:
		LOG.info("Supplying projection layers from %s", projection_dir)
		command.append(f"projectionlayers={str(projection_dir)}")

	# Add categorical layers
	if categorical:
		unique_names = list(dict.fromkeys(categorical))  # preserve order, remove duplicates
		LOG.info("Treating layers as categorical: %s", ", ".join(unique_names))
		for name in unique_names:
			command.append(f"togglelayertype={name}")

	LOG.info("Running MaxEnt: %s", " ".join(command))
	try:
		subprocess.run(command, check=True, cwd=maxent_dir)
	except subprocess.CalledProcessError as exc:
		raise RuntimeError(f"MaxEnt execution failed with exit code {exc.returncode}") from exc

	# After successful run, normalize outputs
	# 1) Rename the primary ASCII grid to <base_dir_name>.asc
	base_name = base_dir.name
	output_dir = (base_dir / MAXENT_OUT_DIR).resolve()
	asc_candidates = [output_dir / "site_projection.asc", output_dir / "site.asc"]
	source_asc = next((p for p in asc_candidates if p.exists()), None)
	if source_asc is None:
		LOG.warning(
			"Expected ASCII output not found. Looked for: %s",
			", ".join(str(p) for p in asc_candidates),
		)
	else:
		target_asc = output_dir / f"{base_name}{_ASCII_SUFFIX}"
		if source_asc.resolve() != target_asc.resolve():
			if target_asc.exists():
				try:
					target_asc.unlink()
				except Exception as exc:  # pragma: no cover - filesystem specifics
					raise RuntimeError(f"Failed to remove existing target file: {target_asc}") from exc
			try:
				source_asc.rename(target_asc)
			except Exception as exc:  # pragma: no cover - filesystem specifics
				raise RuntimeError(
					f"Failed to rename '{source_asc.name}' to '{target_asc.name}'"
				) from exc
			LOG.info("Renamed %s to %s", source_asc.name, target_asc.name)

	# 2) Write projection file as both site.prj (legacy) and <base_dir_name>.prj
	src_prj = (base_dir / CRS_FILENAME).resolve()
	newname_prj = output_dir / f"{base_name}.prj"
	try:
		shutil.copy(src_prj, newname_prj)
		LOG.info("Saved projection files: %s", newname_prj.name)
	except Exception as exc:  # pragma: no cover - filesystem specifics
		raise RuntimeError(f"Failed to write projection files into {output_dir}") from exc
