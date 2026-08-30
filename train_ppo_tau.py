"""
train_ppo_tau.py
----------------
Trains a PPO agent on DroneEnvTau2D (the tau-augmented environment).

Differences compared to train_ppo.py (baseline):
  - Uses DroneEnvTau2D instead of DroneEnv2D
    → observation is 5D: [x, z, vx, vz, tau]
    → tau-regulation shaping reward guides agent toward dτ/dt ≈ -0.5
  - Saves to models_tau/ instead of models/ (to avoid overwriting the baseline)
  - ALWAYS starts fresh (no resume from baseline — different observation space!)

After training, run eval_compare.py to compare this agent with the baseline.

Run from insect_landing/ folder:
    python train_ppo_tau.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from envs import DroneEnvTau2D


# ------------------------------------------------------------------ #
#  Custom callback: prints a one-line progress summary every N steps  #
# ------------------------------------------------------------------ #

class ProgressCallback(BaseCallback):
    """Prints training progress without needing TensorBoard."""

    def __init__(self, print_every: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        if self.num_timesteps % self.print_every == 0 and self.episode_rewards:
            mean_r = np.mean(self.episode_rewards[-50:])
            mean_l = np.mean(self.episode_lengths[-50:])
            print(
                f"  Timestep {self.num_timesteps:>8,d} | "
                f"mean reward (last 50 ep): {mean_r:+7.1f} | "
                f"mean length: {mean_l:5.0f} steps"
            )
        return True


# ------------------------------------------------------------------ #
#  Training configuration                                             #
# ------------------------------------------------------------------ #

TOTAL_TIMESTEPS = 1_000_000
N_ENVS    = 4
MODEL_DIR = "models_tau"
LOG_DIR   = "logs_tau"
MODEL_NAME = "ppo_drone_tau"


# ------------------------------------------------------------------ #
#  Main training function                                             #
# ------------------------------------------------------------------ #

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    print("=" * 65)
    print("  DroneEnvTau2D — PPO Training  (tau-augmented environment)")
    print("=" * 65)
    print(f"  Observation space : 5D  [x, z, vx, vz, tau]")
    print(f"  Tau-regulation    : dtau/dt target = {DroneEnvTau2D.TAU_DOT_TARGET}")
    print(f"  Tau shaping coef  : {DroneEnvTau2D.TAU_SHAPING_COEF}")
    print(f"  Total timesteps   : {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs     : {N_ENVS}")
    print(f"  Model save path   : {MODEL_DIR}/{MODEL_NAME}_final.zip")
    print("=" * 65)

    # --- Training environment ---
    train_env = make_vec_env(
        DroneEnvTau2D,
        n_envs=N_ENVS,
        seed=0,
        wrapper_class=Monitor,
    )

    # VecNormalize: normalise observations (all 5D) and rewards.
    # We ALWAYS start fresh for the tau model — the baseline's vec_normalize.pkl
    # has 4D statistics and cannot be reused here.
    norm_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    if os.path.exists(norm_path):
        print(f"  Resuming normalisation stats from: {norm_path}")
        train_env = VecNormalize.load(norm_path, train_env)
        train_env.norm_reward = True
        train_env.training = True
    else:
        print("  Starting fresh normalisation stats.\n")
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- Evaluation environment ---
    eval_env = make_vec_env(DroneEnvTau2D, n_envs=1, seed=99, wrapper_class=Monitor)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)
    eval_env.obs_rms = train_env.obs_rms

    # --- Load or create model ---
    # The tau model CANNOT resume from the 4D baseline model.
    # Only resume if a previous tau training run exists.
    best_model_path  = os.path.join(MODEL_DIR, "best", "best_model.zip")
    final_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final.zip")

    if os.path.exists(best_model_path):
        print(f"  Resuming tau model from: {best_model_path}\n")
        model = PPO.load(best_model_path, env=train_env, seed=42)
    elif os.path.exists(final_model_path):
        print(f"  Resuming tau model from: {final_model_path}\n")
        model = PPO.load(final_model_path, env=train_env, seed=42)
    else:
        print("  No saved tau model — starting fresh.\n")
        # Same hyperparameters as the baseline for a fair comparison.
        # The only differences are the environment (5D obs) and the tau shaping reward.
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
            policy_kwargs=dict(net_arch=[64, 64]),  # same architecture as baseline
            tensorboard_log=None,
            verbose=0,
            seed=42,
        )

    print(f"Neural network: {model.policy}\n")

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // N_ENVS,
        save_path=MODEL_DIR,
        name_prefix=MODEL_NAME,
        verbose=0,
    )

    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=f"{MODEL_DIR}/best",
        log_path=LOG_DIR,
        eval_freq=20_000 // N_ENVS,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    progress_cb = ProgressCallback(print_every=10_000)

    # --- Train ---
    print("Starting training...\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb, progress_cb],
        progress_bar=False,
    )

    # --- Save final model + normalisation stats ---
    final_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final")
    model.save(final_path)
    train_env.save(norm_path)

    print(f"\nTraining complete.")
    print(f"  Final model  : {final_path}.zip")
    print(f"  Best model   : {MODEL_DIR}/best/best_model.zip")
    print(f"  Norm stats   : {norm_path}")
    print("\nNext step: run  python eval_compare.py  to compare both agents.")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
