#!/usr/bin/env python3
"""Parse dllogger training outputs and create a CSV summary."""

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_dllogger_file(filepath: Path) -> dict | None:
    """Parse a dllogger file and extract training results.

    Args:
        filepath: Path to the dllogger JSON file.

    Returns:
        Dictionary with training results and configuration.
    """
    lines = []
    with filepath.open() as f:
        lines = f.readlines()

    params = {}
    start_timestamp = None
    end_timestamp = None
    best_mae = None
    final_mae = None

    for line in lines:
        line = line.strip()
        if not line.startswith("DLLL "):
            continue

        entry = json.loads(line.removeprefix("DLLL "))

        timestamp = float(entry["timestamp"])
        if start_timestamp is None:
            start_timestamp = timestamp
        end_timestamp = timestamp

        step = entry.get("step")
        data = entry.get("data", {})

        if step == "PARAMETER":
            if params:
                print(f"ERROR: Multiple PARAMETER entries found in {filepath}")
                sys.exit(1)
            params = data

        if "validation MAE" in data:
            final_mae = float(data["validation MAE"])

        if "validation best MAE" in data:
            if best_mae is not None:
                print(f"ERROR: Multiple validation best MAE entries found in {filepath}")
                sys.exit(1)
            best_mae = float(data["validation best MAE"])

    if start_timestamp is None or end_timestamp is None:
        print(f"ERROR: No log lines found in {filepath}")
        sys.exit(1)

    # Calculate total time
    total_time = end_timestamp - start_timestamp

    # Prepare result dictionary
    batch_size_per_gpu = params["batch_size"]
    world_size = params.get("world_size")
    if world_size is None:
        world_size = 8 if "multi" in filepath.stem.split("_") else 1
    result = {
        "mode": "train",
        "batch_size": batch_size_per_gpu * world_size,
        "batch_size_per_gpu": batch_size_per_gpu,
        "world_size": world_size,
        "epochs": params["epochs"],
        "total_time": total_time,
        "final_mae": final_mae,
        "best_mae": best_mae,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parse dllogger training outputs and create a CSV summary."
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
        help="Output CSV file path. Defaults to 'training_results.csv' in the results directory.",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.results_dir / "training_results.csv"

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
    keys = set()

    for filepath in dllogger_files:
        res = parse_dllogger_file(filepath)
        if res:
            results.append(res)
            keys.update(res.keys())

    if not results:
        print("No valid results found.", file=sys.stderr)
        sys.exit(1)

    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
