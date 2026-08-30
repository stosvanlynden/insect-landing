# This file makes the 'envs' folder a Python package.
# Importing DroneEnv2D from here lets other scripts do:
#   from envs import DroneEnv2D
# Importing DroneEnvTau2D gives the tau-augmented version:
#   from envs import DroneEnvTau2D

from envs.drone_env import DroneEnv2D
from envs.drone_env_tau import DroneEnvTau2D
