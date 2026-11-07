"""Pull nflverse tables and cache them as Parquet in data/raw/.

Run from the repo root:
    uv run python -m ml_fantasy_football.data.pull
    uv run python -m ml_fantasy_football.data.pull --tables injuries snap_counts --force
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import nflreadpy
import polars as pl
import yaml

CONFIG_PATH = Path("configs/data.yaml")
RAW_DIR = Path("data/raw")


def pull_table(
    name: str,
    loader: Callable[..., pl.DataFrame],
    seasons: list[int] | None,
    raw_dir: Path = RAW_DIR,
    force: bool = False,
) -> list[str]:
    """Save one table to Parquet and return a list of failure messages.

    With seasons, writes data/raw/<name>/<season>.parquet for each season.
    With seasons=None, writes data/raw/<name>.parquet. Existing files are
    skipped unless force is true. Each file is written to a temporary path
    first, so an interrupted pull never leaves a partial file behind.
    """
    failures = []
    targets = (
        [(season, raw_dir / name / f"{season}.parquet") for season in seasons]
        if seasons is not None
        else [(None, raw_dir / f"{name}.parquet")]
    )
    for season, path in targets:
        label = f"{name} {season}" if season is not None else name
        if path.exists() and not force:
            print(f"skip  {label} (cached)", flush=True)
            continue
        try:
            frame = loader(season) if season is not None else loader()
            if frame.height == 0:
                raise ValueError("loader returned no rows")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(".tmp")
            frame.write_parquet(temp_path)
            temp_path.replace(path)
            print(f"saved {label}: {frame.height:,} rows, {frame.width} columns", flush=True)
        except Exception as error:  # Keep going so one bad table doesn't stop the rest.
            failures.append(f"{label}: {error}")
            print(f"FAIL  {label}: {error}", flush=True)
    return failures


def main() -> int:
    """Pull the configured tables. Returns a non-zero exit code on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", help="Table names to pull (default: all)")
    parser.add_argument("--force", action="store_true", help="Re-download cached files")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    first, last = config["seasons"]["first"], config["seasons"]["last"]
    seasons = list(range(first, last + 1))

    unknown = set(args.tables or []) - set(config["tables"])
    if unknown:
        parser.error(f"unknown tables: {sorted(unknown)}")

    failures = []
    for name, table in config["tables"].items():
        if args.tables and name not in args.tables:
            continue
        loader = getattr(nflreadpy, table["loader"])
        failures += pull_table(
            name,
            loader,
            seasons if table["per_season"] else None,
            force=args.force,
        )

    if failures:
        print(f"\n{len(failures)} failed:", *failures, sep="\n  ")
        return 1
    print("\nAll tables pulled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
