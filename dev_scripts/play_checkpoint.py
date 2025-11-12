#!/usr/bin/env python3
"""Simple CLI helper to play checkpoints from logs/rsl_rl/g1_reach_direct."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs" / "rsl_rl" / "g1_reach_direct"


def pick_from_list(options: list[str], prompt: str) -> str:
    if not options:
        raise RuntimeError("No options available to choose from.")
    print(prompt)
    for idx, item in enumerate(options, 1):
        print(f"[{idx}] {item}")
    while True:
        choice = input("Enter index: ").strip()
        if not choice.isdigit():
            print("Please enter a number.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(options):
            return options[idx - 1]
        print("Index out of range, try again.")


def main() -> None:
    if not LOG_ROOT.exists():
        raise FileNotFoundError(f"Log root not found: {LOG_ROOT}")

    run_dirs = sorted([p for p in LOG_ROOT.iterdir() if p.is_dir()])
    run_name = pick_from_list([p.name for p in run_dirs], "Select a run directory:")
    run_path = LOG_ROOT / run_name

    checkpoints = sorted(run_path.glob("*.pt"))
    if not checkpoints:
        raise RuntimeError(f"No .pt checkpoints found in {run_path}")

    ckpt_name = pick_from_list([ckpt.name for ckpt in checkpoints], f"Select a checkpoint in {run_name}:")
    ckpt_path = run_path / ckpt_name

    cmd = [
        "./isaaclab.sh",
        "-p",
        "scripts/reinforcement_learning/rsl_rl/play.py",
        "--task",
        "Isaac-G1-Reach-Direct-v0",
        "--num_envs",
        "1",
        "--headless",
        "--enable_cameras",
        "--video",
        "--video_length",
        "500",
        "--checkpoint",
        str(ckpt_path),
    ]

    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
