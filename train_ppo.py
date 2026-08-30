"""
train_ppo.py
------------
Trains a PPO (Proximal Policy Optimization) agent on DroneEnv2D.

What is PPO?
------------
PPO is a popular reinforcement learning algorithm that works well with
continuous action spaces (like our ax, az controls). It learns by:
  1. Running the current policy (neural network) in the environment
  2. Measuring which actions led to high rewards
  3. Updating the network — but not too aggressively (the "proximal" part)
     so that training stays stable.

We use Stable-Baselines3 (SB3) which gives us a ready-to-use PPO
implementation. We only need to plug in our custom environment.

Run from insect_landing/ folder:
    python train_ppo.py

The trained model is saved to:
    models/ppo_drone_final.zip
Training logs (for TensorBoard) go to:
    logs/ppo_drone/
"""

import sys
import os

# Make sure the package root is on the path
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

from envs import DroneEnv2D


# ------------------------------------------------------------------ #
#  Custom callback: prints a one-line summary every N timesteps       #
# ------------------------------------------------------------------ #

class ProgressCallback(BaseCallback):
    """
    Prints training progress to the console so you can follow along
    without needing TensorBoard.
    """

    def __init__(self, print_every: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every
        self.episode_rewards = []   # buffer for completed episode rewards
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # SB3 stores episode info in self.locals["infos"] after each step
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        # Print summary every N timesteps
        if self.num_timesteps % self.print_every == 0 and self.episode_rewards:
            mean_r = np.mean(self.episode_rewards[-50:])   # last 50 episodes
            mean_l = np.mean(self.episode_lengths[-50:])
            print(
                f"  Timestep {self.num_timesteps:>8,d} | "
                f"mean reward (last 50 ep): {mean_r:+7.1f} | "
                f"mean length: {mean_l:5.0f} steps"
            )
        return True  # returning False would stop training early


# ------------------------------------------------------------------ #
#  Training configuration                                             #
# ------------------------------------------------------------------ #

# Total number of environment steps to train for.
# More = better agent, but slower. Start with 200k to get a feel.
# A well-trained agent typically needs 500k–1M steps.
TOTAL_TIMESTEPS = 1_000_000

# Number of parallel environments to collect experience from.
# More envs = faster data collection. 4 is a safe default.
N_ENVS = 4

# Where to save models and logs
MODEL_DIR = "models"
LOG_DIR   = "logs"
MODEL_NAME = "ppo_drone"


# ------------------------------------------------------------------ #
#  Main training function                                             #
# ------------------------------------------------------------------ #

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    print("=" * 60)
    print("  DroneEnv2D — PPO Training")
    print("=" * 60)
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs   : {N_ENVS}")
    print(f"  Model save path : {MODEL_DIR}/{MODEL_NAME}_final.zip")
    print("=" * 60)

    # --- Create the training environment ---
    # make_vec_env wraps N_ENVS copies of our environment in parallel.
    # Monitor wraps each env to track episode rewards and lengths.
    train_env = make_vec_env(
        DroneEnv2D,
        n_envs=N_ENVS,
        seed=0,
        wrapper_class=Monitor,
    )

    # --- VecNormalize: load existing stats if available, else create fresh ---
    norm_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    if os.path.exists(norm_path):
        print(f"  Resuming normalisation stats from: {norm_path}")
        train_env = VecNormalize.load(norm_path, train_env)
        train_env.norm_reward = True
        train_env.training = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- Create a separate evaluation environment ---
    eval_env = make_vec_env(DroneEnv2D, n_envs=1, seed=99, wrapper_class=Monitor)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)

    # --- Load existing model or create a new one ---
    # If a previously trained model exists, we resume from it so we don't
    # throw away progress. Otherwise we start fresh with default hyperparams.
    final_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final.zip")
    best_model_path  = os.path.join(MODEL_DIR, "best", "best_model.zip")

    if os.path.exists(best_model_path):
        print(f"  Resuming from best model: {best_model_path}\n")
        model = PPO.load(best_model_path, env=train_env, seed=42)
    elif os.path.exists(final_model_path):
        print(f"  Resuming from final model: {final_model_path}\n")
        model = PPO.load(final_model_path, env=train_env, seed=42)
    else:
        print("  No saved model found — starting fresh.\n")
        # policy="MlpPolicy": fully-connected neural network
        # Input: 4 state values (x, z, vx, vz)
        # Output: 2 action values (ax, az)
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
            seed=42,
        )

    print(f"\nNeural network architecture: {model.policy}\n")

    # Synchronise the eval env's normalisation statistics with the training env.
    # They start identical; during training only the train_env stats update,
    # so we pass those stats to eval_env before each evaluation via the callback.
    eval_env.obs_rms = train_env.obs_rms

    # --- Callbacks ---

    # Save a checkpoint every 50k steps so you don't lose progress
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // N_ENVS,   # per-env steps
        save_path=MODEL_DIR,
        name_prefix=MODEL_NAME,
        verbose=0,
    )

    # Evaluate the agent every 20k steps and save the best version
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=f"{MODEL_DIR}/best",
        log_path=LOG_DIR,
        eval_freq=20_000 // N_ENVS,
        n_eval_episodes=20,           # run 20 episodes and average the reward
        deterministic=True,           # use the greedy policy for evaluation
        verbose=1,
    )

    # Our custom progress printer
    progress_cb = ProgressCallback(print_every=10_000)

    # --- Train! ---
    print("Starting training...\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb, progress_cb],
        progress_bar=False,    # set True only if tqdm+rich are installed
    )

    # --- Save the final model and normalisation stats ---
    final_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final")
    model.save(final_path)

    # VecNormalize stats (running mean/std) must be saved separately so the
    # eval script can reproduce the same input scaling.
    norm_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    train_env.save(norm_path)

    print(f"\nTraining complete.")
    print(f"  Final model  : {final_path}.zip")
    print(f"  Best model   : {MODEL_DIR}/best/best_model.zip")
    print(f"  Norm stats   : {norm_path}")
    print("\nNext step: run  python eval_agent.py  to see how the agent performs.")

    train_env.close()
    eval_env.close()


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    train()
