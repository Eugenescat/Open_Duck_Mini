from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from sb3_contrib import TQC
from stable_baselines3 import A2C, PPO, SAC

from env_factory import make_bdx_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a trained policy to MP4.")
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
        help="Algorithm to record.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Model seed id.")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to record.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Max steps per episode before force reset.",
    )
    parser.add_argument("--fps", type=int, default=50, help="Output video FPS.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mp4 path. Default: <run-dir>/videos/<algo>_seed_<seed>.mp4",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions. Default is deterministic actions.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["simple", "imitation", "walk"],
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


def to_bgr(frame):
    if frame is None:
        raise RuntimeError("render() returned None; expected RGB frame.")
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    algo = args.algo.lower()
    run_dir = args.run_dir.resolve()
    model_path = run_dir / "models" / algo / f"seed_{args.seed}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if args.output is None:
        video_dir = run_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        output_path = video_dir / f"{algo}_seed_{args.seed}.mp4"
    else:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    env_name = resolve_env_name(run_dir, args.env)
    env = make_bdx_env(render_mode="rgb_array", env_name=env_name)
    model = load_model(algo, model_path, env)

    deterministic = not args.stochastic
    writer = None

    try:
        for _ in range(args.episodes):
            obs, _ = env.reset()
            frame = env.render()
            bgr = to_bgr(frame)
            if writer is None:
                height, width = bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {output_path}")
            writer.write(bgr)

            for _ in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _, terminated, truncated, _ = env.step(action)
                writer.write(to_bgr(env.render()))
                if terminated or truncated:
                    break
    finally:
        if writer is not None:
            writer.release()
        env.close()

    print(str(output_path))


if __name__ == "__main__":
    main()
