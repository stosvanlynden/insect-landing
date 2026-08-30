"""
test_random_agent.py
--------------------
Draait ÉÉN episode van DroneEnv2D met een puur willekeurige agent en
plot daarna het volledige traject, zodat je visueel kan checken of de
natuurkunde en de rewardfunctie logisch aanvoelen.

Uitvoeren vanuit de map insect_landing/:
    python -m tests.test_random_agent
of simpelweg:
    python tests/test_random_agent.py
"""

import sys
import os

# Zorg dat de projectroot (insect_landing/) op het Python-pad staat,
# zodat 'from envs import DroneEnv2D' werkt, ongeacht hoe je dit uitvoert.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from envs import DroneEnv2D


def run_random_episode(seed: int = 42):
    """
    Maakt de omgeving aan, reset hem, en doorloopt dan één episode met
    willekeurige acties uit de actieruimte.

    Returns
    -------
    history : dict met lijsten van geregistreerde waarden per stap
    total_reward : float — som van alle rewards in de episode
    """

    env = DroneEnv2D()
    obs, _ = env.reset(seed=seed)

    # Opslag voor trajectgegevens
    history = {
        "x":      [],
        "z":      [],
        "vx":     [],
        "vz":     [],
        "ax":     [],       # actie: horizontale versnelling
        "az":     [],       # actie: verticale stuwkracht-versnelling
        "reward": [],
        "step":   [],
    }

    total_reward = 0.0
    terminated   = False
    truncated    = False

    while not (terminated or truncated):
        # Een volledig willekeurige actie trekken uit de actieruimte
        action = env.action_space.sample()

        # De omgeving één stap verder zetten
        obs, reward, terminated, truncated, info = env.step(action)

        # Alles registreren om te kunnen plotten
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
    Maakt een figuur met 4 panelen:
      1. 2D-vluchtpad (x vs z)
      2. Hoogte over tijd (z vs step)
      3. Snelheden over tijd (vx en vz vs step)
      4. Reward over tijd
    """

    steps   = history["step"]
    x       = history["x"]
    z       = history["z"]
    vx      = history["vx"]
    vz      = history["vz"]
    rewards = history["reward"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DroneEnv2D — Random Agent Trajectory", fontsize=14, fontweight="bold")

    # ---- Paneel 1: 2D-traject in de ruimte (x vs z) ----
    ax1 = axes[0, 0]
    ax1.plot(x, z, color="steelblue", linewidth=1.5, label="flight path")
    ax1.plot(x[0],  z[0],  "go", markersize=10, label="start")   # groen = start
    ax1.plot(x[-1], z[-1], "rs", markersize=10, label="end")     # rood  = einde
    # Het landingsvlak tekenen
    ax1.axhspan(0, 0.2, xmin=0, xmax=1, color="lightgreen", alpha=0.3, label="ground")
    ax1.axvspan(-DroneEnv2D.LAND_X_THRESHOLD, DroneEnv2D.LAND_X_THRESHOLD,
                alpha=0.15, color="gold", label="target zone")
    ax1.set_xlabel("Horizontal position x (m)")
    ax1.set_ylabel("Height z (m)")
    ax1.set_title("2-D Flight Path")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ---- Paneel 2: hoogte over tijd ----
    ax2 = axes[0, 1]
    ax2.plot(steps, z, color="darkorange", linewidth=1.5)
    ax2.axhline(0, color="green", linewidth=1.2, linestyle="--", label="ground z=0")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Height z (m)")
    ax2.set_title("Height over Time")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # ---- Paneel 3: snelheden over tijd ----
    ax3 = axes[1, 0]
    ax3.plot(steps, vx, color="royalblue",  linewidth=1.5, label="vx (horizontal)")
    ax3.plot(steps, vz, color="firebrick",  linewidth=1.5, label="vz (vertical)")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Velocity (m/s)")
    ax3.set_title("Velocities over Time")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # ---- Paneel 4: rewardsignaal over tijd ----
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
