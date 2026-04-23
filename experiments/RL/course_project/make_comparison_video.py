"""Build a 2x2 comparison MP4 with algorithm labels in the top-left corner of each tile."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
VIDEOS = [
    ("PPO (300k)", ROOT / "runs/main_ppo_sac_300k/videos/ppo_seed_0.mp4"),
    ("SAC (300k)", ROOT / "runs/main_ppo_sac_300k/videos/sac_seed_0.mp4"),
    ("A2C (150k)", ROOT / "runs/supp_a2c_tqc_150k/videos/a2c_seed_0.mp4"),
    ("TQC (150k)", ROOT / "runs/supp_a2c_tqc_150k/videos/tqc_seed_0.mp4"),
]
OUTPUT = ROOT / "figures/four_algo_comparison_labeled.mp4"
MAX_SECONDS = 10
FPS = 50
TILE = 480  # per-panel size in px


def open_frames(path: Path, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != TILE or frame.shape[1] != TILE:
            frame = cv2.resize(frame, (TILE, TILE))
        frames.append(frame)
    cap.release()
    # If this clip is shorter than the longest, pad with the last frame
    if frames and len(frames) < max_frames:
        pad = max_frames - len(frames)
        frames.extend([frames[-1].copy()] * pad)
    return frames


def annotate(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 10, 10 + th
    # Semi-transparent black box
    overlay = out.copy()
    cv2.rectangle(overlay, (x - 6, y - th - 6), (x + tw + 6, y + baseline + 4),
                  (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness,
                cv2.LINE_AA)
    return out


def main() -> None:
    max_frames = MAX_SECONDS * FPS
    clips = [open_frames(p, max_frames) for _, p in VIDEOS]
    n = min(len(c) for c in clips)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out_w, out_h = TILE * 2, TILE * 2
    writer = cv2.VideoWriter(str(OUTPUT),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (out_w, out_h))
    for i in range(n):
        tiles = [annotate(clips[k][i], VIDEOS[k][0]) for k in range(4)]
        top = np.concatenate([tiles[0], tiles[1]], axis=1)
        bot = np.concatenate([tiles[2], tiles[3]], axis=1)
        grid = np.concatenate([top, bot], axis=0)
        writer.write(grid)
    writer.release()
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
