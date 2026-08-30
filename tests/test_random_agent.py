"""
test_random_agent.py
--------------------
Runs ONE episode of DroneEnv2D with a purely random agent and then
plots the full trajectory so you can visually verify that the physics
and reward function look sensible.

Run from the insect_landing/ folder:
    python -m tests.test_random_agent
or simply:
    python tests/test_random_agent.py
"""

import sys
import os

# Make sure the package root (insect_landing/) is on the Python path
# so that 'from envs import DroneEnv2D' works regardless of how you run this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from envs import DroneEnv2D


def run_random_episode(seed: int = 42):
    """
    Creates the environment, resets it, then steps through one episode
    using random actions sampled from the action space.

    Returns
    -------
    history : dict with lists of recorded values per step
    total_reward : float — sum of all rewards in the episode
    """

    env = DroneEnv2D()
    obs, _ = env.reset(seed=seed)

    # Storage for trajectory data
    history = {
        "x":      [],
        "z":      [],
        "vx":     [],
        "vz":     [],
        "ax":     [],       # action: horizontal acceleration
        "az":     [],       # action: vertical thrust acceleration
        "reward": [],
        "step":   [],
    }

    total_reward = 0.0
    terminated   = False
    truncated    = False

    while not (terminated or truncated):
        # Sample a completely random action from the action space
        action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Record everything for plotting
        history["x"].append(info["x"])
        history["z"].append(info["z"])
        history["vx"].append(info["vx"])
        history["vz"].append(info["vz"])
        history["ax"].append(float(action[0]))
        history["az"].append(float(action[1]))
        history["reward"].append(reward)
        history["step"].append(info["step"])

        total_reward += reward

    env.close()

    print(f"Episode finished after {len(history['step'])} steps.")
    print(f"Final position : x={history['x'][-1]:.2f} m,  z={history['z'][-1]:.2f} m")
    print(f"Final velocity : vx={history['vx'][-1]:.2f} m/s,  vz={history['vz'][-1]:.2f} m/s")
    print(f"Total reward   : {total_reward:.2f}")
    print(f"Terminated={terminated},  Truncated={truncated}")

    return history, total_reward


def plot_trajectory(history: dict):
    """
    Produces a 4-panel figure:
      1. 2-D flight path  (x vs z)
      2. Height over time (z vs step)
      3. Velocities over time (vx and vz vs step)
      4. Reward over time
    """

    steps   = history["step"]
    x       = history["x"]
    z       = history["z"]
    vx      = history["vx"]
    vz      = history["vz"]
    rewards = history["reward"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DroneEnv2D — Random Agent Trajectory", fontsize=14, fontweight="bold")

    # ---- Panel 1: 2-D spatial trajectory (x vs z) ----
    ax1 = axes[0, 0]
    ax1.plot(x, z, color="steelblue", linewidth=1.5, label="flight path")
    ax1.plot(x[0],  z[0],  "go", markersize=10, label="start")   # green = start
    ax1.plot(x[-1], z[-1], "rs", markersize=10, label="end")     # red   = end
    # Draw the landing pad target
    ax1.axhspan(0, 0.2, xmin=0, xmax=1, color="lightgreen", alpha=0.3, label="ground")
    ax1.axvspan(-DroneEnv2D.LAND_X_THRESHOLD, DroneEnv2D.LAND_X_THRESHOLD,
                alpha=0.15, color="gold", label="target zone")
    ax1.set_xlabel("Horizontal position x (m)")
    ax1.set_ylabel("Height z (m)")
    ax1.set_title("2-D Flight Path")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ---- Panel 2: Height over time ----
    ax2 = axes[0, 1]
    ax2.plot(steps, z, color="darkorange", linewidth=1.5)
    ax2.axhline(0, color="green", linewidth=1.2, linestyle="--", label="ground z=0")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Height z (m)")
    ax2.set_title("Height over Time")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # ---- Panel 3: Velocities over time ----
    ax3 = axes[1, 0]
    ax3.plot(steps, vx, color="royalblue",  linewidth=1.5, label="vx (horizontal)")
    ax3.plot(steps, vz, color="firebrick",  linewidth=1.5, label="vz (vertical)")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Velocity (m/s)")
    ax3.set_title("Velocities over Time")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # ---- Panel 4: Reward signal over time ----
    ax4 = axes[1, 1]
    cumulative = np.cumsum(rewards)
    ax4.bar(steps, rewards, color="mediumpurple", alpha=0.6, width=0.8, label="step reward")
    ax4.plot(steps, cumulative, color="darkviolet", linewidth=1.5, label="cumulative reward")
    ax4.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Reward")
    ax4.set_title("Reward over Time")
    ax4.legend(fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("random_agent_trajectory.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to: random_agent_trajectory.png")
    plt.show()


# Startpunt

if __name__ == "__main__":
    history, total_reward = run_random_episode(seed=42)
    plot_trajectory(history)
