"""Continue training a saved SB3 model on the same env.

Usage:
    python experiments/RL/course_project/continue_train.py \
        --from-run runs/ppo_walk_1_5M \
        --algo ppo \
        --seed 0 \
        --env walk \
        --timesteps 2000000 \
        --run-name ppo_walk_3_5M_continued
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import TQC
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.monitor import Monitor

from env_factory import make_bdx_env


ALGO_CLASSES = {"ppo": PPO, "sac": SAC, "a2c": A2C, "tqc": TQC}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run", type=Path, required=True,
                        help="Previous run dir containing models/<algo>/seed_<seed>.zip")
    parser.add_argument("--algo", choices=list(ALGO_CLASSES), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="walk")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/RL/course_project/runs"))
    args = parser.parse_args()

    ckpt = args.from_run / "models" / args.algo / f"seed_{args.seed}.zip"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    run_dir = args.output_dir / args.run_name
    model_dir = run_dir / "models" / args.algo
    monitor_dir = run_dir / "monitor"
    tb_dir = run_dir / "tensorboard" / args.algo
    model_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_config.json").write_text(json.dumps({
        "algorithms": [args.algo],
        "seeds": [args.seed],
        "timesteps": args.timesteps,
        "device": args.device,
        "env_name": args.env,
        "continued_from": str(ckpt),
    }, indent=2))

    env = make_bdx_env(render_mode=None, env_name=args.env)
    env = Monitor(env, str(monitor_dir / f"{args.algo}_seed_{args.seed}.csv"))

    cls = ALGO_CLASSES[args.algo]
    model = cls.load(str(ckpt), env=env, device=args.device,
                     tensorboard_log=str(run_dir / "tensorboard"))

    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False,
                progress_bar=True)
    model.save(str(model_dir / f"seed_{args.seed}"))
    env.close()


if __name__ == "__main__":
    main()
