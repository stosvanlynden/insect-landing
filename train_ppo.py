"""
train_ppo.py
------------
Traint een PPO (Proximal Policy Optimization) agent op DroneEnv2D.

Wat is PPO?
-----------
PPO is een populair reinforcement-learningalgoritme dat goed werkt met
continue actieruimtes (zoals onze ax-, az-besturing). Het leert door:
  1. Het huidige beleid (neuraal netwerk) in de omgeving te draaien
  2. Te meten welke acties tot hoge rewards leidden
  3. Het netwerk bij te werken — maar niet te agressief (het "proximal"-deel),
     zodat het trainen stabiel blijft.

We gebruiken Stable-Baselines3 (SB3), dat een kant-en-klare PPO-implementatie
geeft. We hoeven alleen onze eigen omgeving erin te pluggen.

Uitvoeren vanuit de map insect_landing/:
    python train_ppo.py

Het getrainde model wordt opgeslagen in:
    models/ppo_drone_final.zip
Trainingslogs (voor TensorBoard) gaan naar:
    logs/ppo_drone/
"""

import sys
import os

# Zorg dat de projectroot op het pad staat
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


# Aangepaste callback: print elke N stappen een regel samenvatting

class ProgressCallback(BaseCallback):
    """
    Print de trainingsvoortgang naar de console, zodat je kan meekijken
    zonder dat TensorBoard nodig is.
    """

    def __init__(self, print_every: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = print_every
        self.episode_rewards = []   # buffer voor afgeronde episode-rewards
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # SB3 zet episode-info in self.locals["infos"] na elke stap
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        # Elke N tijdstappen een samenvatting printen
        if self.num_timesteps % self.print_every == 0 and self.episode_rewards:
            mean_r = np.mean(self.episode_rewards[-50:])   # laatste 50 episodes
            mean_l = np.mean(self.episode_lengths[-50:])
            print(
                f"  Timestep {self.num_timesteps:>8,d} | "
                f"mean reward (last 50 ep): {mean_r:+7.1f} | "
                f"mean length: {mean_l:5.0f} steps"
            )
        return True  # False teruggeven zou de training vroegtijdig stoppen


# Trainingsconfiguratie

# Totaal aantal omgevingsstappen om op te trainen.
# Meer = betere agent, maar trager. Begin met 200k om een gevoel te krijgen.
# Een goed getrainde agent heeft doorgaans 500k-1M stappen nodig.
TOTAL_TIMESTEPS = 1_000_000

# Aantal parallelle omgevingen om ervaring uit te verzamelen.
# Meer omgevingen = snellere dataverzameling. 4 is een veilige standaard.
N_ENVS = 4

# Waar modellen en logs worden opgeslagen
MODEL_DIR = "models"
LOG_DIR   = "logs"
MODEL_NAME = "ppo_drone"


# Hoofdfunctie voor training

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

    # --- Trainingsomgeving aanmaken ---
    # make_vec_env wikkelt N_ENVS kopieën van onze omgeving parallel in.
    # Monitor wikkelt elke omgeving zodat episode-rewards en -lengtes bijgehouden worden.
    train_env = make_vec_env(
        DroneEnv2D,
        n_envs=N_ENVS,
        seed=0,
        wrapper_class=Monitor,
    )

    # --- VecNormalize: bestaande statistieken laden indien aanwezig, anders vers beginnen ---
    norm_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    if os.path.exists(norm_path):
        print(f"  Resuming normalisation stats from: {norm_path}")
        train_env = VecNormalize.load(norm_path, train_env)
        train_env.norm_reward = True
        train_env.training = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --- Een aparte evaluatie-omgeving aanmaken ---
    eval_env = make_vec_env(DroneEnv2D, n_envs=1, seed=99, wrapper_class=Monitor)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)

    # --- Bestaand model laden of een nieuwe aanmaken ---
    # Als er al een eerder getraind model bestaat, gaan we daarmee verder
    # zodat er geen voortgang verloren gaat. Anders beginnen we vers met
    # standaard hyperparameters.
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
        # policy="MlpPolicy": volledig verbonden neuraal netwerk
        # Input: 4 toestandswaarden (x, z, vx, vz)
        # Output: 2 actiewaarden (ax, az)
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

    # De normalisatiestatistieken van de eval-omgeving synchroniseren met de
    # trainingsomgeving. Ze beginnen identiek; tijdens het trainen worden
    # alleen de statistieken van train_env bijgewerkt, dus we geven die
    # statistieken vóór elke evaluatie door aan eval_env via de callback.
    eval_env.obs_rms = train_env.obs_rms

    # --- Callbacks ---

    # Elke 50k stappen een checkpoint opslaan, zodat er geen voortgang verloren gaat
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // N_ENVS,   # stappen per omgeving
        save_path=MODEL_DIR,
        name_prefix=MODEL_NAME,
        verbose=0,
    )

    # Elke 20k stappen de agent evalueren en de beste versie opslaan
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=f"{MODEL_DIR}/best",
        log_path=LOG_DIR,
        eval_freq=20_000 // N_ENVS,
        n_eval_episodes=20,           # 20 episodes draaien en de reward middelen
        deterministic=True,           # het greedy beleid gebruiken tijdens evaluatie
        verbose=1,
    )

    # Onze eigen voortgangsprinter
    progress_cb = ProgressCallback(print_every=10_000)

    # --- Trainen! ---
    print("Starting training...\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb, progress_cb],
        progress_bar=False,    # alleen True zetten als tqdm+rich geïnstalleerd zijn
    )

    # --- Het eindmodel en de normalisatiestatistieken opslaan ---
    final_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final")
    model.save(final_path)

    # VecNormalize-statistieken (lopend gemiddelde/std) moeten apart worden
    # opgeslagen, zodat het eval-script dezelfde inputschaling kan reproduceren.
    norm_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    train_env.save(norm_path)

    print(f"\nTraining complete.")
    print(f"  Final model  : {final_path}.zip")
    print(f"  Best model   : {MODEL_DIR}/best/best_model.zip")
    print(f"  Norm stats   : {norm_path}")
    print("\nNext step: run  python eval_agent.py  to see how the agent performs.")

    train_env.close()
    eval_env.close()


# Startpunt

if __name__ == "__main__":
    train()
