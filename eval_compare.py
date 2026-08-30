"""
eval_compare.py
---------------
Compares two trained PPO agents side-by-side:
  - BASELINE : trained on DroneEnv2D (4D obs: [x, z, vx, vz])
  - TAU AGENT: trained on DroneEnvTau2D (5D obs: [x, z, vx, vz, τ])

For BOTH agents we compute tau from the trajectory even if the agent did
not observe it during training. This lets us answer the key question:
  "Does the tau agent show more biological tau-regulation (dτ/dt ≈ -0.5)
   compared to the baseline?"

Output:
  1. Side-by-side summary table (success rate, landing speed, etc.)
  2. eval_compare.png — 6-panel figure comparing trajectories and tau profiles

Run from insect_landing/ folder:
    python eval_compare.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import DroneEnv2D, DroneEnvTau2D


# Bestandspaden

BASELINE_MODEL = "models/best/best_model.zip"
BASELINE_NORM  = "models/vec_normalize.pkl"

TAU_MODEL      = "models_tau/best/best_model.zip"
TAU_NORM       = "models_tau/vec_normalize.pkl"

N_EVAL_EPISODES = 30


# Hulpfunctie: tau + tau_dot berekenen uit een traject

def compute_tau_profile(z_list: list, vz_list: list, dt: float = 0.1) -> tuple:
    """
    Compute tau and dτ/dt for each step of a trajectory.

    tau = z / |vz|  (clipped to TAU_MAX when not descending)
    tau_dot = finite-difference approximation of dτ/dt

    Returns (tau_list, tau_dot_list) — same length as input.
    """
    TAU_MAX = 100.0
    tau_list = []

    for z, vz in zip(z_list, vz_list):
        if vz < -0.01 and z > 0.0:
            tau = float(np.clip(z / (-vz), 0.0, TAU_MAX))
        else:
            tau = TAU_MAX
        tau_list.append(tau)

    # Finite-difference for dτ/dt; first point has no previous, set to NaN
    tau_dot_list = [float("nan")]
    for i in range(1, len(tau_list)):
        tau_dot_list.append((tau_list[i] - tau_list[i - 1]) / dt)

    return tau_list, tau_dot_list


# Een episode uitvoeren, volledig traject verzamelen

def run_episode(model, vec_env, env_class) -> dict:
    """
    Run one deterministic episode and return the full trajectory.

    Works for both DroneEnv2D (4D obs) and DroneEnvTau2D (5D obs).
    Raw state is read from the info dict (real physical units, not normalised).
    """
    obs    = vec_env.reset()
    history = {"x": [], "z": [], "vx": [], "vz": [], "step": []}
    total_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        i = info[0]
        history["x"].append(i["x"])
        history["z"].append(i["z"])
        history["vx"].append(i["vx"])
        history["vz"].append(i["vz"])
        history["step"].append(i["step"])
        total_reward += float(reward[0])

        if done[0]:
            break

    x_f  = history["x"][-1]
    z_f  = history["z"][-1]
    vz_f = history["vz"][-1]
    vx_f = history["vx"][-1]

    landed  = z_f <= 0.05
    safe    = (landed
               and abs(x_f)  <= env_class.LAND_X_THRESHOLD
               and abs(vz_f) <= env_class.LAND_VZ_THRESHOLD
               and abs(vx_f) <= env_class.LAND_VX_THRESHOLD)

    # Compute tau profile from raw trajectory
    tau_list, tau_dot_list = compute_tau_profile(history["z"], history["vz"])

    return {
        "history":      history,
        "tau":          tau_list,
        "tau_dot":      tau_dot_list,
        "total_reward": total_reward,
        "safe_landing": safe,
        "n_steps":      len(history["step"]),
        "x_final":      x_f,
        "z_final":      z_f,
        "vz_final":     vz_f,
        "vx_final":     vx_f,
    }


# N episodes evalueren voor een agent

def evaluate_agent(model, vec_env, env_class, label: str, n: int = N_EVAL_EPISODES) -> list:
    """Run n episodes and print per-episode results."""
    print(f"\n{'-' * 60}")
    print(f"  {label}  ({n} episodes)")
    print(f"{'-' * 60}")

    results = []
    for ep in range(n):
        r = run_episode(model, vec_env, env_class)
        results.append(r)
        status = "SAFE " if r["safe_landing"] else "CRASH"
        print(
            f"  Ep {ep+1:>3d}: {status} | "
            f"reward={r['total_reward']:+7.1f} | "
            f"steps={r['n_steps']:>4d} | "
            f"vz={r['vz_final']:+5.2f} m/s"
        )
    return results


# Vergelijkingssamenvatting printen

def print_summary(baseline_results: list, tau_results: list, n: int):
    """Print a side-by-side comparison table."""

    def stats(results):
        rewards   = [r["total_reward"]  for r in results]
        safe      = sum(r["safe_landing"] for r in results)
        vz_lands  = [abs(r["vz_final"])  for r in results if r["z_final"] <= 0.05]
        steps     = [r["n_steps"]        for r in results]
        return safe, rewards, vz_lands, steps

    b_safe, b_rew, b_vz, b_steps = stats(baseline_results)
    t_safe, t_rew, t_vz, t_steps = stats(tau_results)

    print("\n")
    print("=" * 65)
    print("  COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Metric':<30}  {'Baseline':>12}  {'Tau agent':>12}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}")

    print(f"  {'Safe landings':<30}  {b_safe:>8}/{n}    {t_safe:>8}/{n}  ")
    print(f"  {'Success rate':<30}  "
          f"{100*b_safe/n:>10.0f}%   {100*t_safe/n:>10.0f}%  ")
    print(f"  {'Mean reward':<30}  "
          f"{np.mean(b_rew):>12.1f}  {np.mean(t_rew):>12.1f}")
    print(f"  {'Std reward':<30}  "
          f"{np.std(b_rew):>12.1f}  {np.std(t_rew):>12.1f}")
    print(f"  {'Mean episode length (steps)':<30}  "
          f"{np.mean(b_steps):>12.0f}  {np.mean(t_steps):>12.0f}")

    if b_vz:
        print(f"  {'Mean landing speed |vz| (m/s)':<30}  "
              f"{np.mean(b_vz):>12.2f}  ", end="")
    else:
        print(f"  {'Mean landing speed |vz| (m/s)':<30}  {'N/A':>12}  ", end="")
    if t_vz:
        print(f"{np.mean(t_vz):>12.2f}")
    else:
        print(f"{'N/A':>12}")

    print("=" * 65)

    # Tau-regulation quality: mean |dτ/dt - (-0.5)| over descent phases
    def tau_reg_error(results):
        errors = []
        for r in results:
            for td in r["tau_dot"][1:]:   # skip first NaN
                if not np.isnan(td) and not np.isinf(td) and abs(td) < 20:
                    errors.append(abs(td - (-0.5)))
        return np.mean(errors) if errors else float("nan")

    b_err = tau_reg_error(baseline_results)
    t_err = tau_reg_error(tau_results)
    print(f"\n  Mean |dtau/dt - (-0.5)|  (lower = more bio-inspired)")
    print(f"    Baseline  : {b_err:.3f}")
    print(f"    Tau agent : {t_err:.3f}")
    if not np.isnan(b_err) and not np.isnan(t_err):
        improvement = 100 * (b_err - t_err) / b_err if b_err > 0 else 0
        print(f"    Improvement: {improvement:+.1f}%")


# Beste episode kiezen (hoogste reward)

def best_result(results: list) -> dict:
    return max(results, key=lambda r: r["total_reward"])


# Vergelijkingsfiguur plotten

def plot_comparison(baseline_best: dict, tau_best: dict):
    """
    6-panel figure:
      Row 1 (baseline): trajectory, tau(t), dτ/dt(t)
      Row 2 (tau agent): same three panels

    The dτ/dt target (-0.5) is shown as a horizontal reference line.
    A constant dτ/dt = -0.5 is the biological insect-landing signature.
    """
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(
        "Tau-theory comparison: Baseline vs Tau agent\n"
        "Biological target: dτ/dt = −0.5  (grey dashed line)",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors   = {"baseline": "royalblue", "tau": "darkorange"}
    labels   = {"baseline": "Baseline (4D obs)", "tau": "Tau agent (5D obs)"}
    datasets = {"baseline": baseline_best, "tau": tau_best}

    for row, (key, result) in enumerate(datasets.items()):
        h       = result["history"]
        steps   = h["step"]
        x, z    = h["x"], h["z"]
        vz      = h["vz"]
        tau     = result["tau"]
        tau_dot = result["tau_dot"]
        col     = colors[key]
        lbl     = labels[key]
        status  = "SAFE" if result["safe_landing"] else "CRASH"

        # ---- Panel A: 2-D trajectory ----
        ax_a = fig.add_subplot(gs[row, 0])
        n = len(x)
        path_colors = plt.cm.plasma(np.linspace(0, 1, max(n - 1, 1)))
        for i in range(n - 1):
            ax_a.plot(x[i:i+2], z[i:i+2], color=path_colors[i], linewidth=2)
        ax_a.plot(x[0],  z[0],  "go", markersize=8, label="start",  zorder=5)
        ax_a.plot(x[-1], z[-1], "rs", markersize=8, label="end",    zorder=5)
        ax_a.axhspan(0, 0.2, color="lightgreen", alpha=0.3)
        ax_a.axvspan(-DroneEnv2D.LAND_X_THRESHOLD, DroneEnv2D.LAND_X_THRESHOLD,
                     alpha=0.15, color="gold", label="target")
        ax_a.set_xlabel("x (m)")
        ax_a.set_ylabel("z (m)")
        ax_a.set_title(f"{lbl}\n2-D trajectory [{status}]")
        ax_a.legend(fontsize=7)
        ax_a.grid(True, linestyle="--", alpha=0.5)

        # ---- Panel B: tau(t) ----
        ax_b = fig.add_subplot(gs[row, 1])
        # Clip display to a readable range (exclude TAU_MAX filler)
        tau_display = [min(t, 30.0) for t in tau]
        ax_b.plot(steps, tau_display, color=col, linewidth=1.8)
        ax_b.set_xlabel("Step")
        ax_b.set_ylabel("τ (s)")
        ax_b.set_title(f"{lbl}\nτ = z / |vz|  over time")
        ax_b.grid(True, linestyle="--", alpha=0.5)

        # Annotate: if tau decreases linearly → good bio-regulation
        ax_b.annotate("Ideal: linear τ↓", xy=(steps[len(steps)//3], tau_display[len(steps)//3]),
                      fontsize=7, color="grey", ha="center")

        # ---- Panel C: dτ/dt(t) ----
        ax_c = fig.add_subplot(gs[row, 2])
        # Filter outliers for display (huge values at hover transitions)
        td_display = [
            td if (not np.isnan(td) and abs(td) < 10) else float("nan")
            for td in tau_dot
        ]
        ax_c.plot(steps, td_display, color=col, linewidth=1.8, alpha=0.9)
        ax_c.axhline(-0.5, color="grey", linewidth=1.5, linestyle="--",
                     label="target dτ/dt = −0.5")
        ax_c.axhline(-1.0, color="lightcoral", linewidth=1.0, linestyle=":",
                     label="freefall (−1.0)")
        ax_c.set_xlabel("Step")
        ax_c.set_ylabel("dτ/dt  (s/s)")
        ax_c.set_ylim(-5, 3)
        ax_c.set_title(f"{lbl}\ndτ/dt  over time")
        ax_c.legend(fontsize=7)
        ax_c.grid(True, linestyle="--", alpha=0.5)

    save_path = "eval_compare.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {save_path}")
    plt.show()


# Startpunt

def load_model_and_env(model_path: str, norm_path: str, env_class):
    """Load a model + its normalised evaluation environment."""
    if not os.path.exists(model_path):
        return None, None

    model = PPO.load(model_path)

    raw_env = DummyVecEnv([lambda ec=env_class: ec()])

    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, raw_env)
        vec_env.training    = False
        vec_env.norm_reward = False
    else:
        print(f"  WARNING: normalisation stats not found at {norm_path}")
        vec_env = raw_env

    return model, vec_env


if __name__ == "__main__":

    print("=" * 65)
    print("  Tau-theory comparison: Baseline vs Tau agent")
    print("=" * 65)

    # --- Load baseline ---
    print(f"\nLoading baseline model: {BASELINE_MODEL}")
    baseline_model, baseline_env = load_model_and_env(
        BASELINE_MODEL, BASELINE_NORM, DroneEnv2D
    )
    if baseline_model is None:
        print(f"  ERROR: baseline model not found at {BASELINE_MODEL}")
        print("  Run  python train_ppo.py  first.")
        sys.exit(1)

    # --- Load tau agent ---
    print(f"Loading tau model     : {TAU_MODEL}")
    tau_model, tau_env = load_model_and_env(
        TAU_MODEL, TAU_NORM, DroneEnvTau2D
    )
    if tau_model is None:
        print(f"  ERROR: tau model not found at {TAU_MODEL}")
        print("  Run  python train_ppo_tau.py  first.")
        sys.exit(1)

    # --- Evaluate both agents ---
    baseline_results = evaluate_agent(baseline_model, baseline_env, DroneEnv2D,
                                      "BASELINE (4D obs, no tau shaping)",
                                      n=N_EVAL_EPISODES)

    tau_results      = evaluate_agent(tau_model, tau_env, DroneEnvTau2D,
                                      "TAU AGENT (5D obs, tau-regulation shaping)",
                                      n=N_EVAL_EPISODES)

    # --- Print summary ---
    print_summary(baseline_results, tau_results, N_EVAL_EPISODES)

    # --- Plot comparison ---
    b_best = best_result(baseline_results)
    t_best = best_result(tau_results)

    print(f"\nBaseline best episode : reward={b_best['total_reward']:+.1f}, "
          f"steps={b_best['n_steps']}, safe={b_best['safe_landing']}")
    print(f"Tau agent best episode: reward={t_best['total_reward']:+.1f}, "
          f"steps={t_best['n_steps']}, safe={t_best['safe_landing']}")

    plot_comparison(b_best, t_best)

    baseline_env.close()
    tau_env.close()
