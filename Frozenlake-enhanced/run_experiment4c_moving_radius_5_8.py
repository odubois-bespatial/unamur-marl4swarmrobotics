"""
run_all_configs.py
Runs all 4 communication mode configs sequentially.
Equivalent to the bash loop:
    for config in config_standalone config_centralized config_proximity config_scout:
        python frozen_lake_marl_competitive_agent.py \
            --config $config --mode both \
            --train_episodes 1000 --test_episodes 200 \
            --goal_mode moving
"""

import subprocess
import sys
import time
import datetime

# ── Parameters ────────────────────────────────────────────────────────────────

CONFIGS       = ["config_proximity_radius_5", "config_proximity_radius_8",  "config_scout_radius_5","config_scout_radius_8"]
MODE          = "both"          # train | run | both
TRAIN_EPISODES = 1000
TEST_EPISODES  = 200
GOAL_MODE      = "moving"        # fixed | random | moving

# ── Run ───────────────────────────────────────────────────────────────────────

def run_config(config: str) -> bool:
    """Run one config and return True if successful."""
    cmd = [
        sys.executable, "frozen_lake_marl_competitive_agent.py",
        "--config",          config,
        "--mode",            MODE,
        "--train_episodes",  str(TRAIN_EPISODES),
        "--test_episodes",   str(TEST_EPISODES),
        "--goal_mode",       GOAL_MODE,
    ]

    print(f"\n{'='*60}")
    print(f"  Config : {config}")
    print(f"  Mode   : {MODE}  |  Train: {TRAIN_EPISODES} ep  |  Test: {TEST_EPISODES} ep")
    print(f"  Goal   : {GOAL_MODE}")
    print(f"  CMD    : {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  [{config}] {status} — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return result.returncode == 0


if __name__ == "__main__":
    overall_start = time.time()
    print(f"\nStarting batch run — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configs: {CONFIGS}")

    results = {}
    for config in CONFIGS:
        ok = run_config(config)
        results[config] = "OK" if ok else "FAILED"

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE — {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*60}")
    for config, status in results.items():
        print(f"  {status:6s}  {config}")
    print()