from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate markdown summary from episodes.csv")
    parser.add_argument(
        "--episodes-csv",
        type=Path,
        required=True,
        help="Path to evaluation/episodes.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown output path.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "algo": row["algo"],
                    "seed": int(row["seed"]),
                    "episode_return": float(row["episode_return"]),
                    "episode_len": float(row["episode_len"]),
                    "mean_forward_speed": float(row["mean_forward_speed"]),
                }
            )
    return rows


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.episodes_csv)

    by_algo = defaultdict(list)
    for row in rows:
        by_algo[row["algo"]].append(row)

    lines = []
    lines.append("# PPO vs SAC Results")
    lines.append("")
    lines.append("| Algo | Episode Return | Episode Length | Mean Forward Speed |")
    lines.append("|---|---:|---:|---:|")

    for algo in sorted(by_algo):
        sub = by_algo[algo]
        ret = [x["episode_return"] for x in sub]
        leng = [x["episode_len"] for x in sub]
        spd = [x["mean_forward_speed"] for x in sub]
        lines.append(
            f"| {algo.upper()} | {fmt(np.mean(ret), np.std(ret))} | {fmt(np.mean(leng), np.std(leng))} | {fmt(np.mean(spd), np.std(spd))} |"
        )

    output_text = "\n".join(lines) + "\n"

    if args.output is not None:
        args.output.write_text(output_text)

    print(output_text)


if __name__ == "__main__":
    main()
