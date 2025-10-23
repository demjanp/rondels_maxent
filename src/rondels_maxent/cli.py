from __future__ import annotations

import logging
import argparse
from pathlib import Path
from typing import Optional

from .ingest import stage_archeodata, stage_enviro
from .process import run_model

def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="rondels_maxent",
		description="Calculate MaxEnt SDM for Neolithic rondels",
	)
	p.add_argument("--input", type=str, required=True, help="Input gpkg dataset")
	p.add_argument("--rules", type=str, required=True, help="Input rules JSON")
	p.add_argument("--enviro", type=str, required=True, help="Environmental data directory")
	p.add_argument("--output", type=str, required=True, help="Output directory")
	p.add_argument("--maxent", type=str, required=True, help="maxent.jar directory")
	p.add_argument("--ul", nargs=2, type=float, required=True, metavar=("UL_LON", "UL_LAT"))
	p.add_argument("--lr", nargs=2, type=float, required=True, metavar=("LR_LON", "LR_LAT"))
	p.add_argument("--min_cell_size", type=int, default=-1, required=False, help="Minimum cell size for background rasters in meters (-1 = automatic)")
	return p

def _ensure_dir(p: Path) -> Path:
	path = Path(p)
	path.mkdir(parents=True, exist_ok=True)
	return path

def main(argv: Optional[list[str]] = None) -> int:
	
	logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
	
	ns = _build_parser().parse_args(argv)
	
	in_path = Path(ns.input)
	rules_path = Path(ns.rules)
	enviro_dir = Path(ns.enviro)
	out_dir = _ensure_dir(Path(ns.output))
	maxent_dir = Path(ns.maxent)
	upper_left, lower_right = tuple(ns.ul), tuple(ns.lr)
	min_cell_size = None if ns.min_cell_size <= 0 else ns.min_cell_size

	stage_archeodata(in_path, rules_path, out_dir)
	enviro_layers = stage_enviro(enviro_dir, out_dir, upper_left, lower_right, min_cell_size)
	run_model(enviro_layers, maxent_dir)
	
	return 0

if __name__ == "__main__":
	raise SystemExit(main())
