from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register, registry

ENV_IDS = {
    "simple": "BDXCourseProjectSimple-v0",
    "imitation": "BDXCourseProjectImitation-v0",
    "walk": "BDXCourseProjectWalk-v0",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@contextlib.contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def make_bdx_env(render_mode: str | None = None, env_name: str = "simple") -> gym.Env:
    """Build the Open Duck Mini environment used by legacy RL scripts.

    The original env uses relative paths. We temporarily switch cwd to the
    legacy script folder so the XML model path resolves consistently.
    """
    if env_name not in ENV_IDS:
        raise ValueError(
            f"Unknown env_name '{env_name}'. Valid options: {sorted(ENV_IDS.keys())}"
        )

    legacy_rl_dir = _repo_root() / "experiments" / "RL" / "new"
    course_project_dir = _repo_root() / "experiments" / "RL" / "course_project"
    legacy_rl_dir_str = str(legacy_rl_dir)
    course_project_dir_str = str(course_project_dir)
    if legacy_rl_dir_str not in sys.path:
        sys.path.insert(0, legacy_rl_dir_str)
    if course_project_dir_str not in sys.path:
        sys.path.insert(0, course_project_dir_str)

    env_id = ENV_IDS[env_name]
    if env_id not in registry:
        if env_name == "simple":
            entry_point = "simple_env:BDXEnv"
        elif env_name == "walk":
            entry_point = "imitation_env_walk:BDXEnv"
        else:
            entry_point = "imitation_env:BDXEnv"
        register(
            id=env_id,
            entry_point=entry_point,
        )

    with _temporary_cwd(legacy_rl_dir):
        env = gym.make(env_id, render_mode=render_mode)
    return env
