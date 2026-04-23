"""Export TensorBoard scalar curves (ep_rew_mean, ep_len_mean) as PNG/PDF plots.

Usage:
    python experiments/RL/course_project/plot_learning_curves.py \
        --run-dir experiments/RL/course_project/runs/main_ppo_sac_300k \
        --output experiments/RL/course_project/runs/main_ppo_sac_300k/learning_curves.pdf

For each algo subdir under <run-dir>/tensorboard/<algo>/, we aggregate seeds
(ALGO_1, ALGO_2, ...) as mean ± std and plot one line per algorithm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(event_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    files = list(event_dir.glob("events.out.tfevents.*"))
    if not files:
        return None
    acc = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    events = acc.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def aggregate_algo(algo_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    seed_dirs = sorted([d for d in algo_dir.iterdir() if d.is_dir()])
    series = []
    common_steps = None
    for sd in seed_dirs:
        result = load_scalar(sd, tag)
        if result is None:
            continue
        steps, values = result
        if common_steps is None:
            common_steps = steps
            series.append(values)
        else:
            # Interpolate onto the shortest common axis if misaligned.
            min_len = min(len(common_steps), len(steps))
            common_steps = common_steps[:min_len]
            series = [s[:min_len] for s in series]
            series.append(values[:min_len])
    if not series:
        return None
    stacked = np.stack(series, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return common_steps, mean, std


def plot(run_dir: Path, tag: str, ylabel: str, output: Path) -> None:
    tb_root = run_dir / "tensorboard"
    algo_dirs = sorted([d for d in tb_root.iterdir() if d.is_dir()])
    fig, ax = plt.subplots(figsize=(6, 4))
    any_plotted = False
    for algo_dir in algo_dirs:
        result = aggregate_algo(algo_dir, tag)
        if result is None:
            print(f"skip {algo_dir.name}: no data for {tag}")
            continue
        steps, mean, std = result
        label = algo_dir.name.upper()
        ax.plot(steps, mean, label=label, linewidth=1.6)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
        any_plotted = True
    if not any_plotted:
        print(f"No data found for tag={tag}; skipping plot.")
        plt.close(fig)
        return
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"saved {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=None,
                        help="Output path prefix (no extension). Defaults to <run-dir>/learning_curves")
    args = parser.parse_args()

    prefix = args.output_prefix or (args.run_dir / "learning_curves")
    plot(args.run_dir, "rollout/ep_rew_mean", "Episode return", Path(f"{prefix}_return.pdf"))
    plot(args.run_dir, "rollout/ep_rew_mean", "Episode return", Path(f"{prefix}_return.png"))
    plot(args.run_dir, "rollout/ep_len_mean", "Episode length", Path(f"{prefix}_length.pdf"))
    plot(args.run_dir, "rollout/ep_len_mean", "Episode length", Path(f"{prefix}_length.png"))


if __name__ == "__main__":
    main()
