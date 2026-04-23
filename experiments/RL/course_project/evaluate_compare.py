from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sb3_contrib import TQC
from stable_baselines3 import A2C, PPO, SAC

from env_factory import make_bdx_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained checkpoints and export per-episode metrics."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory created by train_compare.py",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Evaluation episodes per model.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Max environment steps per episode.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["simple", "imitation"],
        help="Environment variant. If omitted, read from run_config.json (fallback simple).",
    )
    return parser.parse_args()


def resolve_env_name(run_dir: Path, cli_env_name: str | None) -> str:
    if cli_env_name is not None:
        return cli_env_name
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return config.get("env_name", "simple")
    return "simple"


def load_model(algo: str, model_path: Path, env):
    if algo == "ppo":
        return PPO.load(str(model_path), env=env)
    if algo == "sac":
        return SAC.load(str(model_path), env=env)
    if algo == "a2c":
        return A2C.load(str(model_path), env=env)
    if algo == "tqc":
        return TQC.load(str(model_path), env=env)
    raise ValueError(f"Unsupported algo: {algo}")


def mean_forward_speed(env) -> float:
    # base linear velocity x component in world frame.
    return float(env.unwrapped.data.body("base").cvel[3])


def evaluate_model(model, env, episodes: int, max_steps: int, deterministic: bool):
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_return = 0.0
        ep_steps = 0
        speed_samples = []

        while not done and not truncated and ep_steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_steps += 1
            speed_samples.append(mean_forward_speed(env))

        rows.append(
            {
                "episode": ep,
                "episode_return": ep_return,
                "episode_len": ep_steps,
                "mean_forward_speed": float(np.mean(speed_samples))
                if speed_samples
                else 0.0,
            }
        )

    return rows


def write_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algo",
                "seed",
                "episode",
                "episode_return",
                "episode_len",
                "mean_forward_speed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = {}
    for algo in sorted({row["algo"] for row in rows}):
        sub = [r for r in rows if r["algo"] == algo]
        summary[algo] = {
            "return_mean": float(np.mean([r["episode_return"] for r in sub])),
            "return_std": float(np.std([r["episode_return"] for r in sub])),
            "len_mean": float(np.mean([r["episode_len"] for r in sub])),
            "len_std": float(np.std([r["episode_len"] for r in sub])),
            "speed_mean": float(np.mean([r["mean_forward_speed"] for r in sub])),
            "speed_std": float(np.std([r["mean_forward_speed"] for r in sub])),
            "num_episodes": len(sub),
        }
    return summary


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    env_name = resolve_env_name(run_dir, args.env)
    model_root = run_dir / "models"
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for algo_dir in sorted(model_root.glob("*")):
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name
        for model_path in sorted(algo_dir.glob("seed_*.zip")):
            seed = int(model_path.stem.replace("seed_", ""))
            env = make_bdx_env(render_mode=None, env_name=env_name)
            model = load_model(algo, model_path, env)

            rows = evaluate_model(
                model,
                env,
                episodes=args.episodes,
                max_steps=args.max_steps,
                deterministic=args.deterministic,
            )

            for row in rows:
                row["algo"] = algo
                row["seed"] = seed
                all_rows.append(row)

            env.close()

    csv_path = eval_dir / "episodes.csv"
    write_csv(csv_path, all_rows)

    summary = summarize(all_rows)
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
