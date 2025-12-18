#!/usr/bin/env python3
"""Parse dllogger benchmark outputs and create a CSV summary.

This script recursively searches a results directory for dllogger JSON files,
extracts the benchmark performance metrics from the last line of each file,
and outputs a CSV with one row per experiment.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_dllogger_file(filepath: Path) -> dict | None:
    """Parse a dllogger file and extract benchmark results from the last line.

    Args:
        filepath: Path to the dllogger JSON file.

    Returns:
        Dictionary with benchmark results.
    """
    # Get the last non-empty line
    last_line = None
    with filepath.open() as f:
        for line in reversed(f.readlines()):
            line = line.strip()
            if line:
                last_line = line
                break

    if not last_line:
        raise RuntimeError(f"No non-empty lines found")

    return json.loads(last_line.removeprefix("DLLL "))["data"]


def main():
    parser = argparse.ArgumentParser(
        description="Parse dllogger benchmark outputs and create a CSV summary."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to the results directory containing dllogger JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV file path. Defaults to 'processed_results.csv' in the results directory.",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.results_dir / "processed_results.csv"

    if not args.results_dir.exists():
        print(f"Error: Directory {args.results_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    dllogger_files = sorted(args.results_dir.rglob("dllogger*.json"))

    if not dllogger_files:
        print(
            f"Error: No dllogger*.json files found in {args.results_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(dllogger_files)} dllogger files.", file=sys.stderr)

    results = []
    for filepath in dllogger_files:
        try:
            results.append(parse_dllogger_file(filepath))
        except Exception as e:
            print(f"Failed to parse {filepath}: {e}", file=sys.stderr)
            sys.exit(1)

    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
