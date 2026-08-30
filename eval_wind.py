"""
eval_wind.py
------------
Experiment C: zero-shot wind robustness evaluation.

Both agents were trained WITHOUT wind.  Here we evaluate them under
increasing wind disturbances to test which agent degrades more gracefully.

Wind model: each step, a random horizontal acceleration in
            [-wind_force, +wind_force] m/s² is added to the agent's
            commanded ax.  This simulates atmospheric turbulence.

Loads all seeds (0, 1, 2) for each agent and evaluates 50 episodes
per wind level per seed.  Results are saved to:
    results/wind_robustness.csv

Run from insect_landing/ folder:
    python eval_wind.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import DroneEnv2D, DroneEnvTau2D


# Configuratie

SEEDS          = [0, 1, 2]
WIND_LEVELS    = [0.0, 0.5, 1.0, 1.5, 2.0]   # m/s² disturbance magnitude
N_EPISODES     = 50                            # per seed per wind level

BASELINE_MODEL_TEMPLATE = os.path.join("models",     "seed_{seed}", "best", "best_model.zip")
BASELINE_NORM_TEMPLATE  = os.path.join("models",     "seed_{seed}", "vec_normalize.pkl")
TAU_MODEL_TEMPLATE      = os.path.join("models_tau", "seed_{seed}", "coeff_0p10", "best", "best_model.zip")
TAU_NORM_TEMPLATE       = os.path.join("models_tau", "seed_{seed}", "coeff_0p10", "vec_normalize.pkl")

OUTPUT_DIR  = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "wind_robustness.csv")


# Hulpfunctie: N episodes uitvoeren, geeft succespercentage + gemiddelde landings-vz terug

def evaluate_wind(model, vec_env, env_class, n_episodes: int, seed_offset: int = 0):
    """
    Run n_episodes with a fixed env seed for reproducibility.
    Returns (success_rate, mean_touchdown_vz).
    """
    successes   = []
    vz_landings = []

    obs = vec_env.reset()
    episodes_done = 0

    while episodes_done < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = vec_env.step(action)
        i = info[0]

        if done[0]:
            x_f  = i["x"]
            z_f  = i["z"]
            vz_f = i["vz"]
            vx_f = i["vx"]

            landed = z_f <= 0.05
            safe = (landed
                    and abs(x_f)  <= env_class.LAND_X_THRESHOLD
                    and abs(vz_f) <= env_class.LAND_VZ_THRESHOLD
                    and abs(vx_f) <= env_class.LAND_VX_THRESHOLD)

            successes.append(int(safe))
            if landed:
                vz_landings.append(abs(vz_f))

            obs = vec_env.reset()
            episodes_done += 1

    success_rate = float(np.mean(successes))
    mean_vz = float(np.mean(vz_landings)) if vz_landings else float("nan")
    return success_rate, mean_vz


# Hoofdprogramma

def load_model_with_norm(model_path, norm_path, env_class, wind_force):
    """Load a trained model with a wind-disturbance eval environment."""
    if not os.path.exists(model_path):
        return None, None

    model   = PPO.load(model_path)

    # Create environment with the desired wind level
    if env_class == DroneEnv2D:
        raw_env = DummyVecEnv([lambda wf=wind_force: DroneEnv2D(wind_force=wf)])
    else:
        raw_env = DummyVecEnv([lambda wf=wind_force: DroneEnvTau2D(wind_force=wf)])

    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, raw_env)
        vec_env.training    = False
        vec_env.norm_reward = False
    else:
        print(f"    WARNING: normalisation stats not found at {norm_path}")
        vec_env = raw_env

    return model, vec_env


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  Experiment C: Wind Robustness Evaluation")
    print(f"  Seeds: {SEEDS}  |  Wind levels: {WIND_LEVELS} m/s2")
    print(f"  Episodes per condition: {N_EPISODES}")
    print("=" * 65)

    rows = []  # will be written to CSV

    for agent_label, model_tmpl, norm_tmpl, env_class in [
        ("baseline", BASELINE_MODEL_TEMPLATE, BASELINE_NORM_TEMPLATE, DroneEnv2D),
        ("tau",      TAU_MODEL_TEMPLATE,      TAU_NORM_TEMPLATE,      DroneEnvTau2D),
    ]:
        print(f"\nAgent: {agent_label.upper()}")

        for seed in SEEDS:
            model_path = model_tmpl.format(seed=seed)
            norm_path  = norm_tmpl.format(seed=seed)

            if not os.path.exists(model_path):
                print(f"  Seed {seed}: model not found at {model_path} -- skipping")
                continue

            print(f"  Seed {seed}: {model_path}")

            for wind in WIND_LEVELS:
                model, vec_env = load_model_with_norm(
                    model_path, norm_path, env_class, wind
                )
                if model is None:
                    continue

                sr, mean_vz = evaluate_wind(model, vec_env, env_class, N_EPISODES)
                vec_env.close()

                print(f"    wind={wind:.1f} m/s2  |  "
                      f"success={sr:.0%}  mean_vz={mean_vz:.3f} m/s")

                rows.append({
                    "agent":        agent_label,
                    "seed":         seed,
                    "wind_force":   wind,
                    "success_rate": sr,
                    "mean_vz":      mean_vz,
                    "n_episodes":   N_EPISODES,
                })

    # Write CSV
    if rows:
        fieldnames = ["agent", "seed", "wind_force", "success_rate", "mean_vz", "n_episodes"]
        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to: {OUTPUT_FILE}")
    else:
        print("\nNo results to save -- check that training has been completed first.")

    print("\nNext step: run  python analyse.py  to generate all figures.")
