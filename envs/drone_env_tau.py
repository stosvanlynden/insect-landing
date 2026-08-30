"""
DroneEnvTau2D — tau-augmented drone landing environment.

Biologically inspired extension of DroneEnv2D based on tau-theory:

    τ (tau) = z / |vz|  =  estimated time-to-contact (seconds)

Insects (honeybees, hoverflies, dragonflies) regulate approach so that
dτ/dt stays constant at approximately -0.5 (Lee 1976; Wagner 1982).
This is called "tau-coupling" or "tau-regulation".

    Why does this work?
    -------------------
    If dτ/dt = κ (constant) and τ₀ = z₀/|vz₀|, then:
      - τ(t) = τ₀ + κ·t
      - Landing occurs when τ = 0, i.e., at t_land = τ₀/|κ|
      - At that moment, vz → 0 as well  (because z → 0 and τ → 0 together)
      - Result: a SOFT landing — speed goes to zero as height goes to zero

    The required thrust for tau-regulation (derived from d/dt of τ = z/|vz|):
      az_net = (κ + 1) · vz² / z     (for κ = -0.5: az_net = 0.5 · vz²/z)
    This is always positive (upward braking), confirming the strategy is safe.

What this environment adds relative to DroneEnv2D:
    1. tau as a 5th observation — the agent directly perceives time-to-contact
    2. Tau-regulation shaping reward — penalises deviation from dτ/dt = -0.5

The landing physics and reward structure are IDENTICAL to DroneEnv2D,
so any difference in agent behaviour is attributable purely to the
tau observation and tau-shaping signal.

Usage:
    from envs import DroneEnvTau2D

    env = DroneEnvTau2D()
    obs, info = env.reset()
    # obs has shape (5,): [x, z, vx, vz, tau]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.drone_env import DroneEnv2D


class DroneEnvTau2D(DroneEnv2D):
    """
    Extends DroneEnv2D with tau-based observation and tau-regulation reward shaping.

    Observation space (5D):
        [x, z, vx, vz, tau]
        tau = z / |vz| clipped to [0, TAU_MAX]

    Extra reward signal (mid-flight only, when descending):
        shaping = -TAU_SHAPING_COEF * min((dτ/dt - TAU_DOT_TARGET)², MAX_ERROR_SQ)

    All other rewards and termination conditions are inherited from DroneEnv2D.
    """

    # ------------------------------------------------------------------ #
    #  Tau-specific constants                                              #
    # ------------------------------------------------------------------ #

    # Maximum tau value — clips when drone is not actively descending.
    # 100 s is an arbitrary "very far away" value; VecNormalize will scale it.
    TAU_MAX: float = 100.0

    # The biological target: insects maintain dτ/dt ≈ -0.5
    # -1.0 would mean constant-speed descent (freefall); -0.5 means deceleration
    TAU_DOT_TARGET: float = -0.5

    # Weight for the tau-regulation shaping reward.
    # Kept small compared to the time penalty (-0.3/step) so the safe-landing
    # incentive remains dominant; tau shaping is a secondary biological guide.
    TAU_SHAPING_COEF: float = 0.10

    # Maximum squared tau-error allowed in the penalty (clips outliers).
    # Without clipping a single step with huge tau_error can overwhelm other signals.
    MAX_TAU_ERROR_SQ: float = 4.0

    # Tau shaping is only applied during meaningful descent (not near the ground
    # where the landing reward takes over, and not when barely moving).
    TAU_MIN_HEIGHT:   float = 0.5    # (m) don't shape below this height
    TAU_MIN_VZ_MAG:   float = 0.10   # (m/s) only shape when descending at ≥ this speed

    # ------------------------------------------------------------------ #
    #  Constructor                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        render_mode=None,
        wind_force: float = 0.0,
        tau_shaping_coef: float = None,   # None = use class default (0.10)
        tau_dot_target:   float = None,   # None = use class default (-0.5)
    ):
        # Initialise the parent DroneEnv2D (sets up 4D obs space, action space, etc.)
        # Also passes wind_force through to the parent.
        super().__init__(render_mode, wind_force=wind_force)

        # Allow overriding the class-level defaults via constructor.
        # This is used by train_all.py for the sensitivity sweep.
        if tau_shaping_coef is not None:
            self.TAU_SHAPING_COEF = float(tau_shaping_coef)
        if tau_dot_target is not None:
            self.TAU_DOT_TARGET = float(tau_dot_target)

        # Override observation space to add the tau dimension
        # [x, z, vx, vz, tau]
        obs_low  = np.array(
            [-self.X_LIMIT, 0.0, -20.0, -20.0, 0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [ self.X_LIMIT, self.Z_MAX, 20.0, 20.0, self.TAU_MAX],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Store tau from the previous step to compute d-tau/dt
        self._tau_prev: float = self.TAU_MAX

    # ------------------------------------------------------------------ #
    #  Helper: compute tau from current state                             #
    # ------------------------------------------------------------------ #

    def _compute_tau(self, z: float, vz: float) -> float:
        """
        tau = z / |vz|  (time-to-contact in seconds).

        Returns TAU_MAX when the drone is not meaningfully descending:
          - vz >= -TAU_MIN_VZ_MAG  (ascending or hovering → tau undefined)
          - z  <= 0                (already on the ground)
        This avoids division-by-zero and keeps the observation bounded.
        """
        if vz >= -self.TAU_MIN_VZ_MAG or z <= 0.0:
            return self.TAU_MAX
        tau = z / (-vz)   # -vz > 0 because vz < 0 when descending
        return float(np.clip(tau, 0.0, self.TAU_MAX))

    # ------------------------------------------------------------------ #
    #  reset() — extend parent reset with tau initialisation             #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        # Parent reset gives us a 4D observation and sets self.state
        obs_4d, info = super().reset(seed=seed, options=options)

        # Compute tau for the initial state
        _, z0, _, vz0 = self.state
        tau0 = self._compute_tau(z0, vz0)
        self._tau_prev = tau0

        # Append tau to the observation
        obs_5d = np.append(obs_4d, tau0).astype(np.float32)

        info["tau"] = tau0
        return obs_5d, info

    # ------------------------------------------------------------------ #
    #  step() — extend parent step with tau observation and shaping       #
    # ------------------------------------------------------------------ #

    def step(self, action: np.ndarray):
        # Run the parent physics + reward.
        # After this call, self.state holds the NEW (x, z, vx, vz).
        obs_4d, reward, terminated, truncated, info = super().step(action)

        # Read the updated state
        x_new, z_new, vx_new, vz_new = self.state

        # Compute new tau
        tau_new = self._compute_tau(z_new, vz_new)

        # ---- Tau-regulation shaping reward ----
        # Applied only during active, meaningful descent (not at touchdown and
        # not when barely moving vertically).
        if (not terminated
                and not truncated
                and vz_new < -self.TAU_MIN_VZ_MAG
                and z_new  >  self.TAU_MIN_HEIGHT):

            # Approximate dτ/dt using a finite-difference over one time step
            tau_dot = (tau_new - self._tau_prev) / self.DT

            # How far is dτ/dt from the biological target (-0.5)?
            tau_error = tau_dot - self.TAU_DOT_TARGET

            # Quadratic penalty clipped to avoid single huge steps from dominating
            tau_shaping = -self.TAU_SHAPING_COEF * min(tau_error ** 2, self.MAX_TAU_ERROR_SQ)
            reward += tau_shaping

        # Compute tau_dot BEFORE updating _tau_prev (uses the old value)
        tau_dot_info = (tau_new - self._tau_prev) / self.DT if not terminated else 0.0

        # Update stored tau for the next step's finite-difference
        self._tau_prev = tau_new

        # Append tau to the info dict so eval scripts can plot it
        info["tau"]     = tau_new
        info["tau_dot"] = tau_dot_info

        # Return 5D observation: [x, z, vx, vz, tau]
        obs_5d = np.append(obs_4d, tau_new).astype(np.float32)

        return obs_5d, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  render() — extend parent with tau information                      #
    # ------------------------------------------------------------------ #

    def render(self):
        x, z, vx, vz = self.state
        tau = self._compute_tau(z, vz)
        print(
            f"Step {self.step_count:>4d} | "
            f"x={x:+7.2f} m  z={z:6.2f} m  "
            f"vx={vx:+6.2f} m/s  vz={vz:+6.2f} m/s  "
            f"τ={tau:6.2f} s"
        )
