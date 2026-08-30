"""
eval_agent.py
-------------
Loads a trained PPO model and runs it in DroneEnv2D to evaluate performance.

What this script does:
  1. Loads the saved model (best or final)
  2. Runs N_EVAL_EPISODES episodes using the trained policy
  3. Prints statistics (success rate, mean reward, mean landing speed)
  4. Plots the trajectory of the BEST episode (highest reward)

Run from insect_landing/ folder:
    python eval_agent.py

Make sure you have trained a model first:
    python train_ppo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import DroneEnv2D


# Configuratie

# Which model to load — prefer the best model saved by EvalCallback
MODEL_PATH = "models/best/best_model.zip"
FALLBACK_MODEL_PATH = "models/ppo_drone_final.zip"

# VecNormalize statistics (must match what was used during training)
NORM_PATH = "models/vec_normalize.pkl"

# How many episodes to evaluate
N_EVAL_EPISODES = 30

# Whether to print each step (useful for debugging, noisy for many episodes)
RENDER_STEPS = False


# Een episode uitvoeren met de getrainde agent

def run_episode(model, vec_env, deterministic: bool = True) -> dict:
    """
    Runs one full episode using a VecNormalize-wrapped environment.

    VecNormalize normalises the observations before feeding them to the model,
    exactly as was done during training. The raw (unnormalised) state is read
    from info so the plots show real physical units.

    deterministic=True  → agent always picks the action with highest probability
                           (the greedy policy — used for evaluation)
    deterministic=False → agent samples from the distribution (exploration)
    """

    obs = vec_env.reset()   # VecEnv returns obs directly (no info tuple)
    history = {
        "x": [], "z": [], "vx": [], "vz": [],
        "ax": [], "az": [], "reward": [], "step": [],
    }
    total_reward = 0.0

    while True:
        # Ask the model what action to take given the NORMALISED observation
        action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, done, info = vec_env.step(action)

        # info is a list of dicts (one per env); we only have 1 env here
        i = info[0]

        history["x"].append(i["x"])
        history["z"].append(i["z"])
        history["vx"].append(i["vx"])
        history["vz"].append(i["vz"])
        history["ax"].append(float(action[0][0]))
        history["az"].append(float(action[0][1]))
        history["reward"].append(float(reward[0]))
        history["step"].append(i["step"])

        total_reward += float(reward[0])

        if done[0]:
            break

    # Classify the outcome
    x_final  = history["x"][-1]
    z_final  = history["z"][-1]
    vz_final = history["vz"][-1]
    vx_final = history["vx"][-1]

    landed      = z_final <= 0.05
    near_target = abs(x_final)  <= DroneEnv2D.LAND_X_THRESHOLD
    slow_vz     = abs(vz_final) <= DroneEnv2D.LAND_VZ_THRESHOLD
    slow_vx     = abs(vx_final) <= DroneEnv2D.LAND_VX_THRESHOLD
    safe        = landed and near_target and slow_vz and slow_vx

    return {
        "history":      history,
        "total_reward": total_reward,
        "safe_landing": safe,
        "n_steps":      len(history["step"]),
        "x_final":      x_final,
        "z_final":      z_final,
        "vz_final":     vz_final,
        "vx_final":     vx_final,
    }


# Evalueren over N episodes en statistieken printen

def evaluate(model, vec_env, n_episodes: int = N_EVAL_EPISODES):

    results = []

    print(f"Running {n_episodes} evaluation episodes...\n")

    for ep in range(n_episodes):
        result = run_episode(model, vec_env, deterministic=True)
        results.append(result)

        status = "SAFE" if result["safe_landing"] else "CRASH"
        print(
            f"  Ep {ep+1:>3d}: {status} | "
            f"reward={result['total_reward']:+7.1f} | "
            f"steps={result['n_steps']:>4d} | "
            f"x={result['x_final']:+5.2f}m  "
            f"vz={result['vz_final']:+5.2f}m/s"
        )

    # Aggregate statistics
    rewards      = [r["total_reward"]  for r in results]
    safe_count   = sum(r["safe_landing"] for r in results)
    landing_vzs  = [abs(r["vz_final"])  for r in results if r["z_final"] <= 0.05]

    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Episodes evaluated  : {n_episodes}")
    print(f"  Safe landings       : {safe_count} / {n_episodes}  "
          f"({100*safe_count/n_episodes:.0f}%)")
    print(f"  Mean reward         : {np.mean(rewards):+.1f}  "
          f"(std: {np.std(rewards):.1f})")
    print(f"  Best reward         : {np.max(rewards):+.1f}")
    print(f"  Worst reward        : {np.min(rewards):+.1f}")
    if landing_vzs:
        print(f"  Mean landing speed  : {np.mean(landing_vzs):.2f} m/s "
              f"(threshold: {DroneEnv2D.LAND_VZ_THRESHOLD} m/s)")
    print("=" * 55)

    # Return the episode with the highest total reward for plotting
    best_result = max(results, key=lambda r: r["total_reward"])
    return results, best_result


# Traject van de beste episode plotten

def plot_best_episode(result: dict):

    h       = result["history"]
    steps   = h["step"]
    x, z    = h["x"], h["z"]
    vx, vz  = h["vx"], h["vz"]
    rewards = h["reward"]
    az      = h["az"]

    status = "SAFE LANDING" if result["safe_landing"] else "CRASH"
    title  = f"DroneEnv2D — Trained PPO Agent  [{status}]"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ---- Panel 1: 2-D spatial trajectory ----
    ax1 = axes[0, 0]
    # Colour the path by time (early = blue, late = red)
    n = len(x)
    colors = plt.cm.plasma(np.linspace(0, 1, n))
    for i in range(n - 1):
        ax1.plot(x[i:i+2], z[i:i+2], color=colors[i], linewidth=2)

    ax1.plot(x[0],  z[0],  "go", markersize=10, label="start",     zorder=5)
    ax1.plot(x[-1], z[-1], "rs", markersize=10, label="end",        zorder=5)
    ax1.axhspan(0, 0.2, color="lightgreen", alpha=0.3, label="ground")
    ax1.axvspan(-DroneEnv2D.LAND_X_THRESHOLD, DroneEnv2D.LAND_X_THRESHOLD,
                alpha=0.15, color="gold", label="target zone")
    ax1.set_xlabel("Horizontal position x (m)")
    ax1.set_ylabel("Height z (m)")
    ax1.set_title("2-D Flight Path  (colour = time, blue→red)")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ---- Panel 2: Height over time ----
    ax2 = axes[0, 1]
    ax2.plot(steps, z, color="darkorange", linewidth=1.8)
    ax2.axhline(0, color="green", linewidth=1.2, linestyle="--", label="ground z=0")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Height z (m)")
    ax2.set_title("Height over Time")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # ---- Panel 3: Velocities over time ----
    ax3 = axes[1, 0]
    ax3.plot(steps, vx, color="royalblue", linewidth=1.8, label="vx (horizontal)")
    ax3.plot(steps, vz, color="firebrick", linewidth=1.8, label="vz (vertical)")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax3.axhline(-DroneEnv2D.LAND_VZ_THRESHOLD, color="firebrick",
                linewidth=1.0, linestyle="--", alpha=0.5, label="vz crash threshold")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Velocity (m/s)")
    ax3.set_title("Velocities over Time")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # ---- Panel 4: Thrust over time ----
    # (more interesting than raw reward for a trained agent)
    ax4 = axes[1, 1]
    ax4.plot(steps, az, color="mediumpurple", linewidth=1.8, label="az thrust")
    ax4.axhline(DroneEnv2D.GRAVITY, color="gray", linewidth=1.0,
                linestyle="--", label=f"hover thrust ({DroneEnv2D.GRAVITY} m/s²)")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Thrust az (m/s²)")
    ax4.set_title("Thrust Command over Time")
    ax4.legend(fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = "eval_best_episode.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.show()


# Startpunt

if __name__ == "__main__":

    # Load the trained model
    if os.path.exists(MODEL_PATH):
        print(f"Loading best model from: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH)
    elif os.path.exists(FALLBACK_MODEL_PATH):
        print(f"Best model not found. Loading final model: {FALLBACK_MODEL_PATH}")
        model = PPO.load(FALLBACK_MODEL_PATH)
    else:
        print("ERROR: No trained model found.")
        print("Run  python train_ppo.py  first.")
        sys.exit(1)

    # --- Build a normalised eval environment ---
    # The agent was trained on normalised observations, so we MUST apply the
    # same normalisation here. We load the running mean/std stats that were
    # saved at the end of training.
    raw_env = DummyVecEnv([lambda: DroneEnv2D()])

    if os.path.exists(NORM_PATH):
        print(f"Loading normalisation stats from: {NORM_PATH}")
        vec_env = VecNormalize.load(NORM_PATH, raw_env)
        vec_env.training = False      # do NOT update stats during evaluation
        vec_env.norm_reward = False   # show real reward, not normalised
    else:
        print("WARNING: normalisation stats not found — results may be poor.")
        vec_env = raw_env

    # Run evaluation and collect results
    results, best = evaluate(model, vec_env, n_episodes=N_EVAL_EPISODES)

    print(f"\nBest episode: reward={best['total_reward']:+.1f}, "
          f"steps={best['n_steps']}, safe={best['safe_landing']}")

    # Plot the best episode
    plot_best_episode(best)
    vec_env.close()
