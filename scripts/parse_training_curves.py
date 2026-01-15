#!/usr/bin/env python3
"""Parse dllogger training outputs to get a training curve."""

import argparse
from collections import defaultdict
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

    if not lines:
        return None

    results = defaultdict(dict)
    for line in lines:
        line = line.strip()
        if not line.startswith("DLLL "):
            continue

        entry = json.loads(line.removeprefix("DLLL "))

        step = entry["step"]
        data = entry["data"]

        if step == "PARAMETER" or step == []:
            continue

        step = int(step)

        if "validation MAE" in data:
            results[step]["validation_mae"] = float(data["validation MAE"])

        if "train loss" in data:
            results[step]["train_loss"] = float(data["train loss"])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Parse dllogger training outputs to get a training curve."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to the results directory containing dllogger JSON files.",
    )

    args = parser.parse_args()

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

    for filepath in dllogger_files:
        res = parse_dllogger_file(filepath)
        if not res:
            print(f"Warning: No data found in {filepath}", file=sys.stderr)
            continue
        res = dict(sorted(res.items()))
        output_file = filepath.with_name(filepath.stem.replace("dllogger", "curve") + ".csv")
        with output_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "validation_mae"])
            writer.writeheader()
            for step, data in res.items():
                row = {"step": step, "train_loss": data["train_loss"], "validation_mae": data.get("validation_mae", "")}
                writer.writerow(row)


if __name__ == "__main__":
    main()
