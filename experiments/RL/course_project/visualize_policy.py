from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer
import json
from sb3_contrib import TQC
from stable_baselines3 import A2C, PPO, SAC

from env_factory import make_bdx_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained policy in MuJoCo.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory created by train_compare.py",
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["ppo", "sac", "a2c", "tqc"],
        help="Algorithm to visualize.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Model seed id.")
    parser.add_argument(
        "--episodes", type=int, default=3, help="How many episodes to play."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Max steps per episode before force reset.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions. Default is deterministic actions.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=5.0,
        help="Seconds to keep viewer open after playback finishes.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["simple", "imitation"],
        help="Environment variant. If omitted, read from run_config.json (fallback simple).",
    )
    return parser.parse_args()


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


def resolve_env_name(run_dir: Path, cli_env_name: str | None) -> str:
    if cli_env_name is not None:
        return cli_env_name
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        return config.get("env_name", "simple")
    return "simple"


def main() -> None:
    args = parse_args()

    model_path = (
        args.run_dir.resolve() / "models" / args.algo.lower() / f"seed_{args.seed}.zip"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    env_name = resolve_env_name(args.run_dir.resolve(), args.env)

    # Use MuJoCo native viewer directly to avoid Gymnasium overlay compatibility
    # issues between specific gymnasium/mujoco versions.
    env = make_bdx_env(render_mode=None, env_name=env_name)
    model = load_model(args.algo.lower(), model_path, env)
    base_env = env.unwrapped
    viewer = mujoco.viewer.launch_passive(base_env.model, base_env.data)

    deterministic = not args.stochastic
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            ep_return = 0.0
            for _ in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                viewer.sync()
                time.sleep(base_env.model.opt.timestep)
                if terminated or truncated:
                    break
            print(f"episode={ep} return={ep_return:.3f}")
        if args.hold_seconds > 0:
            print(f"holding viewer for {args.hold_seconds:.1f}s ...")
            time.sleep(args.hold_seconds)
    finally:
        viewer.close()
        env.close()


if __name__ == "__main__":
    main()
