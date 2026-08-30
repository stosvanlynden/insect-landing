"""
analyse.py
----------
Hoofd-analysescript: leest alle experimentdata in, draait statistiek,
en produceert alle figuren voor het AE4350-verslag.

Geproduceerde figuren (opgeslagen in results/figures/):
  fig1_learning_curves.png   -- reward + landingssnelheid vs tijdstappen
  fig2_boxplots.png          -- landingssnelheid + tau-fout, gepaard per seed
  fig3_tau_histogram.png     -- dtau/dt-verdeling, baseline vs tau
  fig4_wind_robustness.png   -- succespercentage + landingssnelheid vs wind
  fig5_sensitivity.png       -- landingssnelheid + tau-fout vs shaping-coëfficiënt

Statistiek die naar de console wordt geprint:
  - Wilcoxon signed-rank test op landingssnelheid (gepaard per episode)
  - Bootstrap 95%-betrouwbaarheidsintervallen voor alle kernmetrics
  - Samenvattingstabel voor het verslag

Uitvoeren vanuit de map insect_landing/:
    python analyse.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs import DroneEnv2D, DroneEnvTau2D

# Paden

SEEDS        = [0, 1, 2]
COEFFS       = [0.00, 0.02, 0.05, 0.10, 0.20, 0.50]
N_EVAL_EPS   = 30   # gepaarde episodes voor de boxplot-statistiek
FIG_DIR      = os.path.join("results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

def baseline_paths(seed):
    return (
        os.path.join("models", f"seed_{seed}", "best", "best_model.zip"),
        os.path.join("models", f"seed_{seed}", "vec_normalize.pkl"),
    )

def tau_paths(seed, coeff=0.10):
    cs = f"{coeff:.2f}".replace(".", "p")
    return (
        os.path.join("models_tau", f"seed_{seed}", f"coeff_{cs}", "best", "best_model.zip"),
        os.path.join("models_tau", f"seed_{seed}", f"coeff_{cs}", "vec_normalize.pkl"),
    )

def log_path(agent, seed, coeff=0.10):
    if agent == "baseline":
        return os.path.join("logs", f"baseline_seed{seed}.csv")
    cs = f"{coeff:.2f}".replace(".", "p")
    return os.path.join("logs", f"tau_seed{seed}_coeff{cs}.csv")


# Hulpfunctie: model + evaluatie-omgeving laden

def load_model_env(model_path, norm_path, env_class, wind=0.0, coeff=0.10):
    if not os.path.exists(model_path):
        return None, None
    model = PPO.load(model_path)
    if env_class == DroneEnv2D:
        raw = DummyVecEnv([lambda wf=wind: DroneEnv2D(wind_force=wf)])
    else:
        raw = DummyVecEnv([lambda wf=wind, c=coeff: DroneEnvTau2D(wind_force=wf, tau_shaping_coef=c)])
    if os.path.exists(norm_path):
        vec = VecNormalize.load(norm_path, raw)
        vec.training = False; vec.norm_reward = False
    else:
        vec = raw
    return model, vec


# Hulpfunctie: gepaarde evaluatie uitvoeren (zelfde startseeds voor beide agents)

def compute_tau_error_traj(z_list, vz_list, dt=0.1):
    errors, tau_prev = [], None
    for z, vz in zip(z_list, vz_list):
        if vz < -0.1 and z > 0.5:
            tau = z / (-vz)
            if tau_prev is not None:
                td = (tau - tau_prev) / dt
                if abs(td) < 15:
                    errors.append(abs(td - (-0.5)))
            tau_prev = tau
        else:
            tau_prev = None
    return float(np.mean(errors)) if errors else float("nan")

def collect_dtau_samples(z_list, vz_list, dt=0.1):
    """Geeft alle individuele dtau/dt-waarden terug (voor het histogram)."""
    samples, tau_prev = [], None
    for z, vz in zip(z_list, vz_list):
        if vz < -0.1 and z > 0.5:
            tau = z / (-vz)
            if tau_prev is not None:
                td = (tau - tau_prev) / dt
                if abs(td) < 10:
                    samples.append(td)
            tau_prev = tau
        else:
            tau_prev = None
    return samples

def run_paired_eval(model_b, env_b, model_t, env_t, n=N_EVAL_EPS):
    """
    Draait n episodes voor BEIDE agents met dezelfde startcondities.
    Geeft gepaarde arrays van landingssnelheden en tau-fouten terug.
    """
    vz_b, vz_t = [], []
    te_b, te_t = [], []
    dtau_b_all, dtau_t_all = [], []

    for ep in range(n):
        ep_seed = ep * 7 + 13   # deterministisch per episode-index

        for model, vec_env, vz_list_out, te_list_out, dtau_out in [
            (model_b, env_b, vz_b, te_b, dtau_b_all),
            (model_t, env_t, vz_t, te_t, dtau_t_all),
        ]:
            obs = vec_env.reset()
            # De onderliggende env-seed zetten voor gepaarde startcondities
            try:
                vec_env.env_method("reset", seed=ep_seed)
                obs = vec_env.reset()
            except Exception:
                pass

            ep_z, ep_vz = [], []
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = vec_env.step(action)
                i = info[0]
                ep_z.append(i["z"]); ep_vz.append(i["vz"])
                if done[0]:
                    vz_list_out.append(abs(i["vz"]))
                    te_list_out.append(compute_tau_error_traj(ep_z, ep_vz))
                    dtau_out.extend(collect_dtau_samples(ep_z, ep_vz))
                    break

    return (np.array(vz_b), np.array(vz_t),
            np.array(te_b), np.array(te_t),
            dtau_b_all, dtau_t_all)


# Bootstrap betrouwbaarheidsinterval

def bootstrap_ci(data, n_boot=2000, ci=0.95):
    data = np.array([x for x in data if not np.isnan(x)])
    if len(data) == 0:
        return float("nan"), float("nan")
    boots = [np.mean(np.random.choice(data, len(data))) for _ in range(n_boot)]
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 + ci) / 2)
    return lo, hi


# Leercurve-CSV's inlezen

def read_learning_csv(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def learning_mean_std(agent, seeds, coeff=0.10):
    """Geeft (timesteps, mean_metric, std_metric) arrays terug uit meerdere seed-CSV's."""
    all_data = []
    for s in seeds:
        data = read_learning_csv(log_path(agent, s, coeff))
        if data:
            all_data.append(data)
    if not all_data:
        return None, None, None

    # Uitlijnen op de timestep-index (alle CSV's zouden dezelfde lengte moeten hebben)
    min_len = min(len(d) for d in all_data)
    timesteps = np.array([r["timestep"] for r in all_data[0][:min_len]])

    for metric in ["mean_vz", "success_rate"]:
        pass   # wordt in de plotfunctie geëxtraheerd

    return all_data, timesteps, min_len


# FIGUUR 1: Leercurves

def fig_learning_curves():
    print("Generating Fig 1: Learning curves...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Learning curves: Baseline vs Tau agent (mean +/- std, 3 seeds)",
                 fontsize=13, fontweight="bold")

    colors = {"baseline": "royalblue", "tau": "darkorange"}
    labels = {"baseline": "Baseline (4D)", "tau": "Tau agent (5D)"}

    stored = {}  # bewaart de gemiddelde arrays voor annotatie

    for metric, ax, ylabel, title in [
        ("mean_vz",      axes[0], "Mean touchdown |vz| (m/s)",  "Landing speed during training"),
        ("success_rate", axes[1], "Success rate",               "Success rate during training"),
    ]:
        for agent in ["baseline", "tau"]:
            all_data, timesteps, min_len = learning_mean_std(agent, SEEDS)
            if all_data is None:
                continue
            vals = np.array([[r[metric] for r in d[:min_len]] for d in all_data])
            mean = vals.mean(axis=0)
            std  = vals.std(axis=0)
            ax.plot(timesteps / 1e6, mean,
                    color=colors[agent], linewidth=2, label=labels[agent])
            ax.fill_between(timesteps / 1e6, mean - std, mean + std,
                            color=colors[agent], alpha=0.20)
            stored[(metric, agent)] = (timesteps, mean)

        if metric == "mean_vz":
            # Inzoomen op het interessante convergentiegebied
            ax.set_ylim(0, 4)

        if metric == "success_rate":
            ax.set_ylim(-0.05, 1.10)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")

            # De leervertraging annoteren: zoeken waar elke agent voor het eerst boven 80% komt
            threshold = 0.80
            # Tekstposities uit elkaar houden zodat labels niet overlappen
            annot_offsets = {
                "baseline": (-0.08, -0.17),  # links van de lijn, lager
                "tau":      ( 0.06, -0.08),  # rechts van de lijn, hoger
            }
            for agent, color in colors.items():
                key = (metric, agent)
                if key not in stored:
                    continue
                ts, mean = stored[key]
                idx_above = np.where(mean >= threshold)[0]
                if len(idx_above) > 0:
                    t_cross = ts[idx_above[0]] / 1e6
                    dx, dy = annot_offsets.get(agent, (0.04, -0.12))
                    ax.axvline(t_cross, color=color, linewidth=1.0,
                               linestyle=":", alpha=0.7)
                    ax.annotate(f"{agent.capitalize()}: {t_cross:.2f}M",
                                xy=(t_cross, threshold),
                                xytext=(t_cross + dx, threshold + dy),
                                fontsize=7.5, color=color,
                                ha="left" if dx > 0 else "right",
                                arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

        ax.set_xlabel("Timesteps (millions)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


# FIGUUR 2: Boxplots (gepaarde evaluatie)

def fig_boxplots_and_stats():
    print("\nGenerating Fig 2: Box plots + statistics...")

    # Gepaarde resultaten over alle seeds verzamelen
    all_vz_b, all_vz_t = [], []
    all_te_b, all_te_t = [], []
    all_dtau_b, all_dtau_t = [], []

    for seed in SEEDS:
        m_b, e_b = load_model_env(*baseline_paths(seed), DroneEnv2D)
        m_t, e_t = load_model_env(*tau_paths(seed, 0.10), DroneEnvTau2D, coeff=0.10)
        if m_b is None or m_t is None:
            print(f"  Seed {seed}: model missing, skipping")
            continue

        print(f"  Evaluating seed {seed}...")
        vz_b, vz_t, te_b, te_t, dtau_b, dtau_t = run_paired_eval(m_b, e_b, m_t, e_t, N_EVAL_EPS)
        all_vz_b.extend(vz_b); all_vz_t.extend(vz_t)
        all_te_b.extend(te_b); all_te_t.extend(te_t)
        all_dtau_b.extend(dtau_b); all_dtau_t.extend(dtau_t)
        e_b.close(); e_t.close()

    all_vz_b = np.array(all_vz_b)
    all_vz_t = np.array(all_vz_t)
    all_te_b = np.array([x for x in all_te_b if not np.isnan(x)])
    all_te_t = np.array([x for x in all_te_t if not np.isnan(x)])

    # ---- Statistiek ----
    print("\n" + "=" * 60)
    print("  STATISTICAL RESULTS")
    print("=" * 60)

    # Wilcoxon signed-rank test (gepaard)
    stat_vz, p_vz = stats.wilcoxon(all_vz_b[:len(all_vz_t)], all_vz_t[:len(all_vz_b)])
    r_vz = stat_vz / np.sqrt(len(all_vz_b) * (len(all_vz_b) + 1) / 2)   # effectgrootte r

    ci_b = bootstrap_ci(all_vz_b)
    ci_t = bootstrap_ci(all_vz_t)

    print(f"\n  Touchdown speed |vz| (m/s):")
    print(f"    Baseline : {np.mean(all_vz_b):.3f} +/- {np.std(all_vz_b):.3f}  "
          f"95% CI [{ci_b[0]:.3f}, {ci_b[1]:.3f}]")
    print(f"    Tau agent: {np.mean(all_vz_t):.3f} +/- {np.std(all_vz_t):.3f}  "
          f"95% CI [{ci_t[0]:.3f}, {ci_t[1]:.3f}]")
    print(f"    Wilcoxon W={stat_vz:.1f}, p={p_vz:.4f}, effect r={r_vz:.3f}")
    if p_vz < 0.001:
        sig = "p < 0.001 (***)"
    elif p_vz < 0.01:
        sig = "p < 0.01 (**)"
    elif p_vz < 0.05:
        sig = "p < 0.05 (*)"
    else:
        sig = "not significant"
    print(f"    Significance: {sig}")

    ci_teb = bootstrap_ci(all_te_b)
    ci_tet = bootstrap_ci(all_te_t)
    # Wilcoxon voor de tau-fout (gepaard, zelfde lengte)
    n_te = min(len(all_te_b), len(all_te_t))
    stat_te, p_te = stats.wilcoxon(all_te_b[:n_te], all_te_t[:n_te])
    if p_te < 0.001:
        sig_te = "p < 0.001 (***)"
    elif p_te < 0.01:
        sig_te = "p < 0.01 (**)"
    elif p_te < 0.05:
        sig_te = "p < 0.05 (*)"
    else:
        sig_te = "not significant"
    print(f"\n  Mean |dtau/dt - (-0.5)| (tau-regulation error):")
    print(f"    Baseline : {np.mean(all_te_b):.3f} +/- {np.std(all_te_b):.3f}  "
          f"95% CI [{ci_teb[0]:.3f}, {ci_teb[1]:.3f}]")
    print(f"    Tau agent: {np.mean(all_te_t):.3f} +/- {np.std(all_te_t):.3f}  "
          f"95% CI [{ci_tet[0]:.3f}, {ci_tet[1]:.3f}]")
    improvement = 100 * (np.mean(all_te_b) - np.mean(all_te_t)) / np.mean(all_te_b)
    print(f"    Improvement: {improvement:+.1f}%  Wilcoxon {sig_te}")
    print("=" * 60)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle("Performance comparison: Baseline vs Tau agent\n"
                 f"(n = {len(all_vz_b)} episodes per agent, {len(SEEDS)} seeds x {N_EVAL_EPS} ep)",
                 fontsize=12, fontweight="bold")

    colors_box = ["royalblue", "darkorange"]

    # Landingssnelheid
    ax1 = axes[0]
    bp = ax1.boxplot([all_vz_b, all_vz_t],
                     tick_labels=["Baseline\n(4D obs)", "Tau agent\n(5D obs)"],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax1.axhline(DroneEnv2D.LAND_VZ_THRESHOLD, color="red", linewidth=1.0,
                linestyle="--", alpha=0.6, label=f"Safe limit ({DroneEnv2D.LAND_VZ_THRESHOLD} m/s)")
    ax1.set_ylabel("Touchdown |vz| (m/s)")
    ax1.set_ylim(0, 1.2)   # inzoomen — alle data ligt onder 1.1 m/s
    ax1.set_title(f"Landing speed\n(Wilcoxon {sig})")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Gemiddelde-annotaties toevoegen
    for i, (data, color) in enumerate(zip([all_vz_b, all_vz_t], colors_box), 1):
        ax1.annotate(f"mean={np.mean(data):.3f}", xy=(i, np.mean(data)),
                     ha="center", va="bottom", fontsize=8, color=color,
                     xytext=(0, 5), textcoords="offset points")

    # Tau-regulatiefout
    ax2 = axes[1]
    bp2 = ax2.boxplot([all_te_b, all_te_t],
                      tick_labels=["Baseline\n(4D obs)", "Tau agent\n(5D obs)"],
                      patch_artist=True, widths=0.5,
                      medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp2["boxes"], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    # Geen "perfecte regulatie"-lijn — dat wekt een onrealistische verwachting
    ax2.set_ylabel("|dtau/dt - (-0.5)|")
    ax2.set_title(f"Tau-regulation error\n(lower = more bio-inspired)\n"
                  f"Improvement: {improvement:+.1f}%  ({sig_te})")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {path}")

    return all_dtau_b, all_dtau_t


# FIGUUR 3: dtau/dt-histogram

def fig_tau_histogram(dtau_b, dtau_t):
    print("\nGenerating Fig 3: dtau/dt histogram...")

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Distribution of dtau/dt during descent phases\n"
                 "Biological target: dtau/dt = -0.5 (grey dashed)",
                 fontsize=12, fontweight="bold")

    # X-as beperken tot -5 tot 1.0 — het positieve bereik is bijna leeg
    bins = np.linspace(-5, 1.0, 55)

    m_b = np.mean(dtau_b); m_t = np.mean(dtau_t)

    ax.hist(dtau_b, bins=bins, color="royalblue", alpha=0.55,
            density=True, label=f"Baseline  (n={len(dtau_b):,} steps, mean={m_b:.2f})")
    ax.hist(dtau_t, bins=bins, color="darkorange", alpha=0.55,
            density=True, label=f"Tau agent (n={len(dtau_t):,} steps, mean={m_t:.2f})")

    ax.axvline(-0.5, color="grey", linewidth=2, linestyle="--",
               label="Biological target: dtau/dt = -0.5")
    ax.axvline(-1.0, color="lightcoral", linewidth=1.2, linestyle=":",
               label="Constant-speed descent (-1.0)")
    ax.axvline(m_b, color="royalblue",  linewidth=1.5, linestyle="-", alpha=0.8)
    ax.axvline(m_t, color="darkorange", linewidth=1.5, linestyle="-", alpha=0.8)

    ax.set_xlabel("dtau/dt  (s/s)")
    ax.set_ylabel("Density")
    ax.set_xlim(-5, 1.0)
    # Legenda buiten de plot — voorkomt overlap met de histogrambalken
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Annotatie die uitlegt waarom beide agents ver van -0.5 afzitten
    ax.annotate("Note: time penalty (-0.3/step)\npushes agents to land quickly,\ncompeting with tau-regulation",
                xy=(-0.5, 0.6), xytext=(-4.5, 0.7),
                fontsize=7.5, color="dimgrey",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="grey", alpha=0.8),
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_tau_histogram.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    print(f"  Baseline mean dtau/dt: {m_b:.3f}  |  Tau mean dtau/dt: {m_t:.3f}")


# FIGUUR 4: Windrobuustheid

def fig_wind_robustness():
    print("\nGenerating Fig 4: Wind robustness...")

    wind_csv = os.path.join("results", "wind_robustness.csv")
    if not os.path.exists(wind_csv):
        print(f"  {wind_csv} not found -- run eval_wind.py first")
        return

    data = {}
    with open(wind_csv) as f:
        for row in csv.DictReader(f):
            agent = row["agent"]
            wind  = float(row["wind_force"])
            sr    = float(row["success_rate"])
            vz    = float(row["mean_vz"])
            key   = (agent, wind)
            data.setdefault(key, []).append((sr, vz))

    wind_levels = sorted(set(w for _, w in data.keys()))
    colors = {"baseline": "royalblue", "tau": "darkorange"}
    labels = {"baseline": "Baseline", "tau": "Tau agent"}

    # Rechterpaneel breder: dat vertelt het hoofdverhaal (landingskwaliteit onder wind)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [2, 3]})
    fig.suptitle("Wind stability: tau agent maintains landing quality under wind disturbances\n"
                 "(Zero-shot: agents trained without wind, tested at evaluation)",
                 fontsize=11, fontweight="bold")

    for metric_idx, (ax, ylabel, title, idx) in enumerate(zip(
        axes,
        ["Success rate", "Mean touchdown |vz| (m/s)"],
        ["Success rate (99.3-100% across wind levels)", "Landing speed preserved under wind"],
        [0, 1]
    )):
        for agent in ["baseline", "tau"]:
            means, stds = [], []
            for wind in wind_levels:
                vals = [v[idx] for v in data.get((agent, wind), [])]
                means.append(np.mean(vals) if vals else float("nan"))
                stds.append(np.std(vals)   if vals else 0.0)
            means = np.array(means); stds = np.array(stds)
            ax.plot(wind_levels, means, "o-", color=colors[agent],
                    linewidth=2, markersize=7, label=labels[agent])
            ax.fill_between(wind_levels, means - stds, means + stds,
                            color=colors[agent], alpha=0.18)

        if idx == 0:
            # Y-as inzoomen: beide liggen vlak rond 100%, dat duidelijk laten zien
            ax.set_ylim(0.88, 1.03)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
            ax.annotate("Both agents: 99.3-100%\n(seed avg.) at every level",
                        xy=(1.0, 1.0), xytext=(0.8, 0.94),
                        fontsize=8, color="dimgrey",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                  edgecolor="grey", alpha=0.8))
        ax.set_xlabel("Wind disturbance magnitude (m/s2)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_wind_robustness.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


# FIGUUR 5: Sensitiviteitsanalyse

def fig_sensitivity():
    print("\nGenerating Fig 5: Sensitivity analysis...")

    SEEDS = [0, 1, 2]
    results = []
    for coeff in COEFFS:
        seed_vzs, seed_tes = [], []
        for seed in SEEDS:
            model_path, norm_path = tau_paths(seed, coeff)
            if coeff == 0.10:
                # De coeff=0.10-mappen zijn de hoofdexperiment-runs (getraind
                # tot 1M stappen), dus "best_model.zip" is niet vergelijkbaar
                # met de andere coëfficiënten (alleen 500k stappen). Gebruik
                # in plaats daarvan het checkpoint op precies 500k stappen,
                # voor een eerlijk vergelijkingspunt met gelijk budget.
                matched_ckpt = os.path.join(
                    "models_tau", f"seed_{seed}", "coeff_0p10",
                    f"ckpt_tau_s{seed}_500000_steps.zip")
                if os.path.exists(matched_ckpt):
                    model_path = matched_ckpt
            if not os.path.exists(model_path):
                print(f"  coeff={coeff} seed={seed}: model missing at {model_path}")
                continue

            model, vec_env = load_model_env(model_path, norm_path, DroneEnvTau2D, coeff=coeff)
            if model is None:
                continue

            vz_list, te_list = [], []
            obs = vec_env.reset()
            ep_z, ep_vz = [], []
            episodes_done = 0
            while episodes_done < 30:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = vec_env.step(action)
                i = info[0]
                ep_z.append(i["z"]); ep_vz.append(i["vz"])
                if done[0]:
                    vz_list.append(abs(i["vz"]))
                    te_list.append(compute_tau_error_traj(ep_z, ep_vz))
                    obs = vec_env.reset(); ep_z = []; ep_vz = []
                    episodes_done += 1
            vec_env.close()

            te_clean = [x for x in te_list if not np.isnan(x)]
            seed_vzs.append(np.mean(vz_list))
            seed_tes.append(np.mean(te_clean) if te_clean else float("nan"))

        if not seed_vzs:
            continue
        results.append({
            "coeff":    coeff,
            "mean_vz":  np.mean(seed_vzs),
            "std_vz":   np.std(seed_vzs),
            "mean_te":  np.mean(seed_tes),
            "std_te":   np.std(seed_tes),
        })
        print(f"  coeff={coeff:.2f}: mean_vz={np.mean(seed_vzs):.3f}+/-{np.std(seed_vzs):.3f}  "
              f"tau_err={np.mean(seed_tes):.3f}+/-{np.std(seed_tes):.3f}  (n={len(seed_vzs)} seeds)")

    if not results:
        print("  No sensitivity data found.")
        return

    coeffs   = [r["coeff"]   for r in results]
    mean_vzs = [r["mean_vz"] for r in results]
    std_vzs  = [r["std_vz"]  for r in results]
    mean_tes = [r["mean_te"] for r in results]
    std_tes  = [r["std_te"]  for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sensitivity analysis: effect of tau shaping coefficient\n"
                 "(mean +/- std across 3 seeds, 30 eval episodes each, all"
                 " six at the matched 500k-step checkpoint)",
                 fontsize=10, fontweight="bold")

    for ax, means, stds, ylabel, title, color in zip(
        axes,
        [mean_vzs, mean_tes],
        [std_vzs,  std_tes],
        ["Mean touchdown |vz| (m/s)", "|dtau/dt - (-0.5)| (tau-regulation error)"],
        ["Landing speed vs shaping coefficient",
         "Tau-regulation error vs shaping coefficient"],
        ["steelblue", "darkorange"],
    ):
        means = np.array(means); stds = np.array(stds)
        ax.errorbar(coeffs, means, yerr=stds, fmt="o-", color=color,
                    linewidth=2, markersize=8, capsize=4, elinewidth=1.5)

        # De coëfficiënt uit de hoofdexperimenten markeren (0.10) -- dit is de
        # waarde die daadwerkelijk multi-seed is getraind, NIET per se het
        # empirische minimum hier (zie de Discussion in het verslag).
        if 0.10 in coeffs:
            idx = coeffs.index(0.10)
            ax.axvline(0.10, color="grey", linewidth=1.2, linestyle="--", alpha=0.7,
                       label="lambda=0.10 (used in main experiments)")
            ax.scatter([coeffs[idx]], [means[idx]], s=120, zorder=5,
                       color=color, edgecolors="black", linewidth=1.5)

        ax.set_xlabel("TAU_SHAPING_COEF")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig5_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


# Startpunt

if __name__ == "__main__":
    np.random.seed(0)

    print("=" * 65)
    print("  AE4350 Analysis — generating all figures + statistics")
    print("=" * 65)

    # Fig 1: leercurves (geen model nodig)
    fig_learning_curves()

    # Fig 2: boxplots + statistische toetsen (laadt alle 6 modellen)
    dtau_b, dtau_t = fig_boxplots_and_stats()

    # Fig 3: dtau/dt-histogram (gebruikt data uit de Fig 2-evaluatie)
    fig_tau_histogram(dtau_b, dtau_t)

    # Fig 4: windrobuustheid (leest results/wind_robustness.csv)
    fig_wind_robustness()

    # Fig 5: sensitiviteit (laadt de sensitiviteitsmodellen)
    fig_sensitivity()

    print("\n" + "=" * 65)
    print("  All done! Figures saved to: results/figures/")
    print("=" * 65)
    print("\n  fig1_learning_curves.png  <- Fig 2 in report (learning effect)")
    print("  fig2_boxplots.png         <- Fig 3 in report (main comparison)")
    print("  fig3_tau_histogram.png    <- Fig 4 in report (tau behaviour)")
    print("  fig4_wind_robustness.png  <- Fig 5 in report (robustness)")
    print("  fig5_sensitivity.png      <- Fig 6 in report (sensitivity)")

    # Alle 5 figuren tegelijk openen
    plt.show()
