from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="rondels_maxent",
		description="Calculate MaxEnt SDM for Neolithic rondels",
	)
	p.add_argument("--input", type=str, required=True, help="Input gpkg dataset")
	p.add_argument("--output", type=str, required=True, help="Output directory")
	return p

def _ensure_dir(p: Path) -> Path:
	path = Path(p)
	path.mkdir(parents=True, exist_ok=True)
	return path

def main(argv: Optional[list[str]] = None) -> int:
	
	ns = _build_parser().parse_args(argv)
	
	in_path = Path(ns.input)
	out_dir = _ensure_dir(Path(ns.output))
	
	return 0

if __name__ == "__main__":
	raise SystemExit(main())
