"""
run_experiments.py
------------------
Batch script that runs ALL training experiments sequentially.

Experiments:
  A+B : baseline seeds 0, 1, 2  (1M steps each)  -- performance + learning curves
  A+B : tau seeds 0, 1, 2        (1M steps each)  -- performance + learning curves
  D   : tau sensitivity sweep    (500k steps each) -- coeff 0.0, 0.02, 0.05, 0.20, 0.50
        (coeff 0.10 is already covered by seed 0 above)

Total: 6 x 1M steps + 5 x 500k steps
Estimated time: ~6 x 25min + ~5 x 12min = ~2.5 hours

After this script finishes, run:
    python eval_wind.py      <- robustness experiment (fast, no training)
    python analyse.py        <- all figures + statistics

Run from insect_landing/ folder:
    python run_experiments.py
"""

import subprocess
import sys
import time

# Use the same Python interpreter that is running this script
PYTHON = sys.executable


def run(args: list, description: str):
    """Run one training job and print its progress live."""
    print("\n" + "=" * 65)
    print(f"  STARTING: {description}")
    print("=" * 65 + "\n")
    t0 = time.time()

    result = subprocess.run(
        [PYTHON] + args,
        check=False,           # don't raise on non-zero exit — we'll report it
    )

    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    if result.returncode == 0:
        print(f"\n  Finished in {mins}m {secs}s: {description}")
    else:
        print(f"\n  ERROR (exit {result.returncode}) after {mins}m {secs}s: {description}")

    return result.returncode


if __name__ == "__main__":

    # ---------------------------------------------------------------- #
    # Experiments A + B: multi-seed training (1M steps each)           #
    # ---------------------------------------------------------------- #

    baseline_runs = [
        (["train_all.py", "--agent", "baseline", "--seed", "0"], "Baseline seed 0"),
        (["train_all.py", "--agent", "baseline", "--seed", "1"], "Baseline seed 1"),
        (["train_all.py", "--agent", "baseline", "--seed", "2"], "Baseline seed 2"),
    ]

    tau_runs = [
        (["train_all.py", "--agent", "tau", "--seed", "0", "--coeff", "0.10"], "Tau seed 0 (coeff=0.10)"),
        (["train_all.py", "--agent", "tau", "--seed", "1", "--coeff", "0.10"], "Tau seed 1 (coeff=0.10)"),
        (["train_all.py", "--agent", "tau", "--seed", "2", "--coeff", "0.10"], "Tau seed 2 (coeff=0.10)"),
    ]

    # ---------------------------------------------------------------- #
    # Experiment D: sensitivity sweep (500k steps each)                #
    # coeff 0.10 + seed 0 is already trained above, so skip it.       #
    # ---------------------------------------------------------------- #

    sensitivity_runs = [
        (["train_all.py", "--agent", "tau", "--seed", "0", "--coeff", "0.00", "--steps", "500000"],
         "Sensitivity coeff=0.00 (baseline, no tau shaping)"),
        (["train_all.py", "--agent", "tau", "--seed", "0", "--coeff", "0.02", "--steps", "500000"],
         "Sensitivity coeff=0.02"),
        (["train_all.py", "--agent", "tau", "--seed", "0", "--coeff", "0.05", "--steps", "500000"],
         "Sensitivity coeff=0.05"),
        (["train_all.py", "--agent", "tau", "--seed", "0", "--coeff", "0.20", "--steps", "500000"],
         "Sensitivity coeff=0.20"),
        (["train_all.py", "--agent", "tau", "--seed", "0", "--coeff", "0.50", "--steps", "500000"],
         "Sensitivity coeff=0.50"),
    ]

    all_runs = baseline_runs + tau_runs + sensitivity_runs

    print(f"Total training runs: {len(all_runs)}")
    print("Estimated time: ~2.5 hours\n")
    print("Runs planned:")
    for i, (args, desc) in enumerate(all_runs):
        print(f"  {i+1:>2d}. {desc}")

    print("\nStarting in 3 seconds... (Ctrl+C to abort)\n")
    time.sleep(3)

    errors = []
    for i, (args, desc) in enumerate(all_runs):
        print(f"\n[{i+1}/{len(all_runs)}]", end="")
        rc = run(args, desc)
        if rc != 0:
            errors.append((i + 1, desc, rc))

    print("\n" + "=" * 65)
    print("  ALL RUNS COMPLETE")
    print("=" * 65)
    if errors:
        print(f"  {len(errors)} run(s) failed:")
        for num, desc, rc in errors:
            print(f"    Run {num}: {desc}  (exit code {rc})")
    else:
        print("  All runs completed successfully.")

    print("\nNext steps:")
    print("  python eval_wind.py   <- wind robustness experiment")
    print("  python analyse.py     <- figures + statistics")
