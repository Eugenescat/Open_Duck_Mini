from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from sb3_contrib import TQC
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env_factory import make_bdx_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RL policies for Open Duck Mini and save comparable outputs."
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["ppo", "sac"],
        choices=["ppo", "sac", "a2c", "tqc"],
        help="Algorithms to train.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds to run.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Training timesteps per algorithm per seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device (e.g. auto, cpu, cuda).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="simple",
        choices=["simple", "imitation", "walk"],
        help="Environment variant to train in.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/RL/course_project/runs"),
        help="Root directory where run folders are created.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Run folder name.",
    )
    return parser.parse_args()


def build_model(algo: str, env, seed: int, tensorboard_dir: Path, device: str):
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            seed=seed,
            device=device,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
        )

    if algo == "sac":
        return SAC(
            "MlpPolicy",
            env,
            seed=seed,
            device=device,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=10_000,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )

    if algo == "a2c":
        return A2C(
            "MlpPolicy",
            env,
            seed=seed,
            device=device,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            learning_rate=7e-4,
            gamma=0.99,
            gae_lambda=1.0,
            n_steps=5,
            ent_coef=0.0,
        )

    if algo == "tqc":
        return TQC(
            "MlpPolicy",
            env,
            seed=seed,
            device=device,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=10_000,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )

    raise ValueError(f"Unsupported algo: {algo}")


def save_run_config(
    run_dir: Path,
    algos: Iterable[str],
    seeds: Iterable[int],
    timesteps: int,
    device: str,
    env_name: str,
) -> None:
    config = {
        "algorithms": list(algos),
        "seeds": list(seeds),
        "timesteps": timesteps,
        "device": device,
        "env_name": env_name,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))


def main() -> None:
    args = parse_args()

    run_dir = args.output_dir / args.run_name
    model_dir = run_dir / "models"
    monitor_dir = run_dir / "monitor"
    tensorboard_dir = run_dir / "tensorboard"

    model_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    save_run_config(
        run_dir, args.algos, args.seeds, args.timesteps, args.device, args.env
    )

    for algo in args.algos:
        for seed in args.seeds:
            set_random_seed(seed)
            env = make_bdx_env(render_mode=None, env_name=args.env)
            env = Monitor(env, str(monitor_dir / f"{algo}_seed_{seed}.csv"))

            model = build_model(
                algo=algo,
                env=env,
                seed=seed,
                tensorboard_dir=tensorboard_dir / algo,
                device=args.device,
            )

            model.learn(total_timesteps=args.timesteps, progress_bar=True)

            model_out = model_dir / algo
            model_out.mkdir(parents=True, exist_ok=True)
            model.save(str(model_out / f"seed_{seed}"))
            env.close()


if __name__ == "__main__":
    main()
