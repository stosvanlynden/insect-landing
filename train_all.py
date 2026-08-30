"""
train_all.py
------------
Eén trainingsscript voor Experimenten A, B en D.

Gebruik (vanuit de map insect_landing/):
    python train_all.py --agent baseline --seed 0
    python train_all.py --agent tau      --seed 1
    python train_all.py --agent tau      --seed 0 --coeff 0.20

Argumenten:
    --agent   : 'baseline' of 'tau'
    --seed    : integer seed (0, 1, 2, ...)
    --coeff   : TAU_SHAPING_COEF (alleen voor de tau-agent, standaard 0.10)
    --steps   : totaal aantal trainingsstappen (standaard 1_000_000)

Output:
    models/<agent>/seed_<seed>/best/best_model.zip
    models/<agent>/seed_<seed>/vec_normalize.pkl
    logs/<agent>_seed<seed>_coeff<coeff>.csv   <- leercurves
        kolommen: timestep, success_rate, mean_vz, std_vz, mean_tau_error

De CSV wordt gebruikt door analyse.py om de leercurve-figuren te maken,
en door eval_wind.py om te bepalen welke modellen geladen moeten worden.
"""

import sys
import os
import argparse
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from envs import DroneEnv2D, DroneEnvTau2D


# Hulpfunctie: tau-fout berekenen uit een traject

def compute_tau_error(z_list, vz_list, dt=0.1):
    """
    Gemiddelde |dtau/dt - (-0.5)| over alle daalstappen in één episode.
    tau = z / |vz|, dtau/dt benaderd met eindige verschillen.
    Geeft NaN terug als de drone nooit betekenisvol heeft gedaald.
    """
    errors = []
    tau_prev = None
    for z, vz in zip(z_list, vz_list):
        if vz < -0.1 and z > 0.5:
            tau = z / (-vz)                     # time-to-contact
            if tau_prev is not None:
                tau_dot = (tau - tau_prev) / dt
                if abs(tau_dot) < 15:           # extreme uitschieters filteren
                    errors.append(abs(tau_dot - (-0.5)))
            tau_prev = tau
        else:
            tau_prev = None                     # resetten bij stijgen/hoveren
    return float(np.mean(errors)) if errors else float("nan")


# Metrics-callback: slaat leercurves op naar CSV

class MetricsCallback(BaseCallback):
    """
    Elke `eval_freq` tijdstappen:
      1. Draait `n_eval_episodes` episodes met het deterministische beleid
      2. Berekent success_rate, gemiddelde/std landings-|vz|, gemiddelde tau-fout
      3. Voegt één rij toe aan het CSV-bestand

    Dit levert de leercurve-data voor Experiment B, en een live beeld van
    de trainingsvoortgang zonder TensorBoard.
    """

    def __init__(
        self,
        eval_env,          # door VecNormalize omwikkelde evaluatie-omgeving
        csv_path: str,     # pad naar het CSV-bestand (wordt aangemaakt/aangevuld)
        eval_freq: int = 20_000,
        n_eval_episodes: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.csv_path        = csv_path
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._last_eval_step = 0

        # CSV-header schrijven
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep", "success_rate",
                "mean_vz", "std_vz", "mean_tau_error",
            ])

    def _on_step(self) -> bool:
        # Alleen elke eval_freq stappen evalueren
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        # Normalisatiestatistieken synchroniseren van de trainings- naar de eval-omgeving
        self.eval_env.obs_rms = self.training_env.obs_rms

        # Evaluatie-episodes draaien
        successes   = []
        vz_landings = []
        tau_errors  = []

        obs = self.eval_env.reset()
        ep_z, ep_vz = [], []
        ep_step = 0

        # We draaien de episodes één voor één (DummyVecEnv met 1 omgeving)
        episodes_done = 0
        while episodes_done < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = self.eval_env.step(action)
            i = info[0]
            ep_z.append(i["z"])
            ep_vz.append(i["vz"])

            if done[0]:
                x_f  = i["x"]
                z_f  = i["z"]
                vz_f = i["vz"]
                vx_f = i["vx"]

                landed = z_f <= 0.05
                safe = (landed
                        and abs(x_f)  <= DroneEnv2D.LAND_X_THRESHOLD
                        and abs(vz_f) <= DroneEnv2D.LAND_VZ_THRESHOLD
                        and abs(vx_f) <= DroneEnv2D.LAND_VX_THRESHOLD)

                successes.append(int(safe))
                if landed:
                    vz_landings.append(abs(vz_f))
                tau_errors.append(compute_tau_error(ep_z, ep_vz))

                # Klaarzetten voor de volgende episode
                obs    = self.eval_env.reset()
                ep_z   = []
                ep_vz  = []
                episodes_done += 1

        # Samenvattende statistieken berekenen
        success_rate  = float(np.mean(successes))
        mean_vz       = float(np.mean(vz_landings))   if vz_landings else float("nan")
        std_vz        = float(np.std(vz_landings))    if vz_landings else float("nan")
        tau_valid     = [e for e in tau_errors if not np.isnan(e)]
        mean_tau_err  = float(np.mean(tau_valid))     if tau_valid   else float("nan")

        # Toevoegen aan de CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.num_timesteps,
                f"{success_rate:.4f}",
                f"{mean_vz:.4f}",
                f"{std_vz:.4f}",
                f"{mean_tau_err:.4f}",
            ])

        # Voortgang printen
        print(
            f"  [{self.num_timesteps:>8,d}] "
            f"success={success_rate:.0%}  "
            f"|vz|={mean_vz:.3f} m/s  "
            f"tau_err={mean_tau_err:.3f}"
        )
        return True


# Hoofdfunctie voor training

def train(agent: str, seed: int, coeff: float, total_steps: int):

    # ---- Mappen klaarzetten ----
    if agent == "baseline":
        model_dir = os.path.join("models", f"seed_{seed}")
        log_name  = f"baseline_seed{seed}"
    else:
        coeff_str = f"{coeff:.2f}".replace(".", "p")
        model_dir = os.path.join("models_tau", f"seed_{seed}", f"coeff_{coeff_str}")
        log_name  = f"tau_seed{seed}_coeff{coeff_str}"

    os.makedirs(model_dir,  exist_ok=True)
    os.makedirs("logs",     exist_ok=True)
    csv_path = os.path.join("logs", f"{log_name}.csv")

    print("=" * 65)
    print(f"  Agent : {agent.upper()}   Seed : {seed}   Steps : {total_steps:,}")
    if agent == "tau":
        print(f"  TAU_SHAPING_COEF = {coeff}   TAU_DOT_TARGET = -0.5")
    print(f"  Model dir : {model_dir}")
    print(f"  CSV log   : {csv_path}")
    print("=" * 65)

    # ---- Omgeving-fabriek ----
    if agent == "baseline":
        env_fn      = lambda: DroneEnv2D()
        eval_env_fn = lambda: DroneEnv2D()
    else:
        env_fn      = lambda c=coeff: DroneEnvTau2D(tau_shaping_coef=c)
        eval_env_fn = lambda c=coeff: DroneEnvTau2D(tau_shaping_coef=c)

    # ---- Trainingsomgeving ----
    train_env = make_vec_env(env_fn, n_envs=4, seed=seed, wrapper_class=Monitor)

    norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    if os.path.exists(norm_path):
        print(f"  Resuming normalisation stats from: {norm_path}")
        train_env = VecNormalize.load(norm_path, train_env)
        train_env.norm_reward = True
        train_env.training    = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ---- Evaluatie-omgeving (1 omgeving, geen reward-normalisatie) ----
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)
    eval_env.obs_rms = train_env.obs_rms

    # ---- Model: begint per seed altijd opnieuw (reproduceerbaarheid) ----
    best_path = os.path.join(model_dir, "best", "best_model.zip")
    if os.path.exists(best_path):
        print(f"  Resuming model from: {best_path}\n")
        model = PPO.load(best_path, env=train_env, seed=seed)
    else:
        print("  Starting fresh model.\n")
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[64, 64]),
            tensorboard_log=None,
            verbose=0,
            seed=seed,
        )

    # ---- Callbacks ----
    checkpoint_cb = CheckpointCallback(
        save_freq   = 100_000 // 4,   # elke 100k stappen
        save_path   = model_dir,
        name_prefix = f"ckpt_{agent}_s{seed}",
        verbose=0,
    )

    # De standaard SB3 EvalCallback slaat het beste model op
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=os.path.join("logs", log_name),
        eval_freq=20_000 // 4,
        n_eval_episodes=20,
        deterministic=True,
        verbose=0,           # standaard output onderdrukken; MetricsCallback print zelf
    )

    # Onze eigen CSV-logger
    metrics_cb = MetricsCallback(
        eval_env=eval_env,
        csv_path=csv_path,
        eval_freq=20_000,    # elke 20k tijdstappen in totaal
        n_eval_episodes=20,
    )

    # ---- Trainen ----
    print("Training started...\n")
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        progress_bar=False,
    )

    # ---- Eindmodel + normalisatiestatistieken opslaan ----
    final_path = os.path.join(model_dir, f"final_model")
    model.save(final_path)
    train_env.save(norm_path)

    print(f"\nDone. Saved:")
    print(f"  Best model  : {model_dir}/best/best_model.zip")
    print(f"  Final model : {final_path}.zip")
    print(f"  Norm stats  : {norm_path}")
    print(f"  CSV log     : {csv_path}")

    train_env.close()
    eval_env.close()


# Startpunt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on DroneEnv2D or DroneEnvTau2D")
    parser.add_argument("--agent",  choices=["baseline", "tau"], required=True,
                        help="Which agent to train: 'baseline' (4D obs) or 'tau' (5D obs)")
    parser.add_argument("--seed",   type=int, default=0,
                        help="Random seed for training (default: 0)")
    parser.add_argument("--coeff",  type=float, default=0.10,
                        help="TAU_SHAPING_COEF for tau agent (default: 0.10)")
    parser.add_argument("--steps",  type=int, default=1_000_000,
                        help="Total training timesteps (default: 1_000_000)")
    args = parser.parse_args()

    train(
        agent      = args.agent,
        seed       = args.seed,
        coeff      = args.coeff,
        total_steps= args.steps,
    )
