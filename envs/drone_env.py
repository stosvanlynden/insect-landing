"""
DroneEnv2D — a minimal 2D drone landing environment built with Gymnasium.

Physics model
-------------
The drone is treated as a **point mass** in the vertical plane (x, z).
There is no aerodynamic model; only gravity and two direct accelerations:
  ax  — horizontal acceleration (control input)
  az  — net vertical acceleration  = thrust_accel - gravity

This is intentionally simple so the RL agent has to learn how to
balance thrust against gravity and steer horizontally to the pad.

Coordinate system
-----------------
  x  — horizontal position (m), positive to the right
  z  — height (m), positive upward.  Ground is at z = 0.
  vx — horizontal velocity (m/s)
  vz — vertical velocity   (m/s), negative means descending

Episode end conditions
----------------------
  1. z <= 0          : drone reached the ground (landing or crash)
  2. |x| > x_limit  : drone flew too far sideways
  3. z  > z_limit    : drone flew too high (out of expected range)
  4. step >= max_steps: time limit exceeded

Reward function (designed for safe landing near x=0, z=0)
----------------------------------------------------------
  Safe landing   : large positive reward scaled by how close to target
                   and how low the impact speed is
  Crash landing  : negative penalty
  Thrust penalty : small penalty proportional to |az_control| each step
                   (encourages energy efficiency — insect-inspired frugality)
  Time penalty   : tiny negative reward every step (encourages efficiency)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DroneEnv2D(gym.Env):
    # ------------------------------------------------------------------ #
    #  Class-level constants — physical and episode parameters            #
    # ------------------------------------------------------------------ #

    # Gravity acceleration (m/s²)
    GRAVITY: float = 9.81

    # Simulation time step (s)  — 10 Hz update rate
    DT: float = 0.1

    # Maximum episode length (steps)
    MAX_STEPS: int = 500

    # Spatial limits — episode ends if drone leaves this box
    X_LIMIT: float = 30.0   # (m) horizontal
    Z_MAX:   float = 50.0   # (m) maximum height

    # Action limits (m/s²)
    MAX_AX: float = 5.0     # max horizontal acceleration
    MAX_AZ: float = 20.0    # max upward thrust acceleration
                             # net upward = 20 - 9.81 = 10.19 m/s² — enough to
                             # brake from -10 m/s in ~1 second
                             # note: net vertical accel = az_thrust - GRAVITY

    # Landing success thresholds
    LAND_X_THRESHOLD:  float = 2.0   # (m)   must land within this of x=0
    LAND_VX_THRESHOLD: float = 2.0   # (m/s) max horizontal speed on touch
    LAND_VZ_THRESHOLD: float = 3.0   # (m/s) max downward speed on touch
                                      #        vz is negative when descending

    # Reward weights
    REWARD_SAFE_BASE:   float =  100.0  # max reward for a perfect safe landing
    REWARD_TIME_STEP:   float =   -0.3  # small penalty every step (encourages speed)
    REWARD_THRUST_COEF: float =   -0.01 # small penalty per (m/s²) thrust used

    # --- Graduated landing penalty ---
    # Instead of a fixed -100 crash, the penalty scales with HOW BAD the
    # landing is. This is the key fix for the hovering problem:
    #
    #   Old design: crash = -100, hover 500 steps = -250 → agent prefers crash
    #               but crash = -100 is still worse than hovering 100 steps (-50)
    #               → agent learns to hover 100 steps then accept crash.
    #
    #   New design: bad landing = -5 to -40 (much smaller than hover 500 steps = -150)
    #               → crashing QUICKLY is always better than hovering
    #               → safe landing (= +100) is much better than any crash
    #               → agent is pushed toward safe landing, not toward hovering
    #
    # Penalty formula on landing:
    #   base          : -5.0  (always, for not landing safely)
    #   speed excess  : -3.0 * max(0, |vz| - VZ_THRESHOLD)   per m/s over limit
    #   lateral excess: -2.0 * max(0, |vx| - VX_THRESHOLD)
    #   position exc. : -1.0 * max(0, |x|  - X_THRESHOLD)
    CRASH_BASE:        float = -5.0
    CRASH_VZ_COEF:     float = -3.0
    CRASH_VX_COEF:     float = -2.0
    CRASH_X_COEF:      float = -1.0

    # --- Potential-based reward shaping (Ng et al. 1999) ---
    # F = gamma * Phi(s') - Phi(s)  added to every mid-flight step.
    # Gives gradient toward lower |vz| and lower |x| WITHOUT rewarding hovering.
    GAMMA_SHAPE:  float = 0.99
    PHI_VZ:       float = 2.0    # weight of vertical-speed potential
    PHI_X:        float = 0.5    # weight of horizontal-position potential
    VZ_MAX_SCALE: float = 20.0

    # ------------------------------------------------------------------ #
    #  Constructor                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, render_mode=None, wind_force: float = 0.0):
        super().__init__()

        self.render_mode = render_mode  # placeholder (no live rendering yet)

        # Optional wind disturbance (m/s²).
        # Each step a random horizontal acceleration in [-wind_force, +wind_force]
        # is added to ax.  0.0 = no wind (default, used during training).
        # Set during EVALUATION only to test zero-shot robustness.
        self.wind_force: float = wind_force

        # --- Action space ---
        # Two continuous actions: [ax, az_control]
        #   ax          in [-MAX_AX,  +MAX_AX]   horizontal acceleration
        #   az_control  in [0,        +MAX_AZ]   upward thrust acceleration
        #               (cannot push downward; gravity already does that)
        self.action_space = spaces.Box(
            low  = np.array([-self.MAX_AX, 0.0],        dtype=np.float32),
            high = np.array([ self.MAX_AX, self.MAX_AZ], dtype=np.float32),
        )

        # --- Observation space ---
        # Four continuous state variables: [x, z, vx, vz]
        obs_low  = np.array([-self.X_LIMIT, 0.0,        -20.0, -20.0], dtype=np.float32)
        obs_high = np.array([ self.X_LIMIT, self.Z_MAX,  20.0,  20.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Internal state (initialised properly in reset())
        self.state: np.ndarray = np.zeros(4, dtype=np.float32)
        self.step_count: int = 0

    # ------------------------------------------------------------------ #
    #  reset() — called at the start of every new episode                 #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random from Gymnasium

        # Randomise starting position inside a reasonable box.
        # Heights are kept lower (3-12 m) so the drone has enough altitude
        # to brake with the available thrust before hitting the ground.
        # Initial velocities start near zero so the agent first learns to
        # hover and steer, then generalises to harder starts.
        x0  = self.np_random.uniform(-8.0,  8.0)   # horizontal offset (m)
        z0  = self.np_random.uniform( 3.0, 12.0)   # starting height (m)
        vx0 = self.np_random.uniform(-0.5,  0.5)   # near-zero horizontal drift
        vz0 = self.np_random.uniform(-0.5,  0.2)   # near-zero vertical velocity

        self.state = np.array([x0, z0, vx0, vz0], dtype=np.float32)
        self.step_count = 0

        info = {}
        return self.state.copy(), info

    # ------------------------------------------------------------------ #
    #  step() — advance the simulation by one time step                   #
    # ------------------------------------------------------------------ #

    def step(self, action: np.ndarray):
        # Unpack current state
        x, z, vx, vz = self.state

        # Clip actions to valid range (safety — in case the agent produces NaN)
        ax          = float(np.clip(action[0], -self.MAX_AX, self.MAX_AX))
        az_control  = float(np.clip(action[1],  0.0,          self.MAX_AZ))

        # Add wind disturbance to horizontal acceleration.
        # np_random is seeded by reset(), so wind is reproducible per seed.
        if self.wind_force > 0.0:
            wind = self.np_random.uniform(-self.wind_force, self.wind_force)
            ax = float(np.clip(ax + wind, -self.MAX_AX * 2, self.MAX_AX * 2))

        # Net vertical acceleration: thrust upward minus gravity downward
        az_net = az_control - self.GRAVITY

        # ------ Euler integration (simple but transparent) ------
        # Update velocities first
        vx_new = vx + ax         * self.DT
        vz_new = vz + az_net     * self.DT

        # Update positions using the new velocities (semi-implicit Euler)
        x_new  = x  + vx_new    * self.DT
        z_new  = z  + vz_new    * self.DT

        self.step_count += 1

        # ------ Determine whether episode ends ------
        terminated = False   # natural end (ground contact, boundary)
        truncated  = False   # artificial end (step limit)
        reward     = 0.0

        # Check if drone reached the ground (z <= 0)
        if z_new <= 0.0:
            terminated = True
            z_new = 0.0  # clamp to ground level

            # How much does each dimension exceed the safe landing threshold?
            vz_excess  = max(0.0, abs(vz_new) - self.LAND_VZ_THRESHOLD)
            vx_excess  = max(0.0, abs(vx_new) - self.LAND_VX_THRESHOLD)
            x_excess   = max(0.0, abs(x_new)  - self.LAND_X_THRESHOLD)

            if vz_excess == 0.0 and vx_excess == 0.0 and x_excess == 0.0:
                # ---- Safe landing ----
                # Reward scales with how perfectly centred and gentle the landing is.
                position_score = 1.0 - abs(x_new)  / self.LAND_X_THRESHOLD
                speed_score    = 1.0 - abs(vz_new) / self.LAND_VZ_THRESHOLD
                reward = self.REWARD_SAFE_BASE * (0.5 * position_score + 0.5 * speed_score)
            else:
                # ---- Graduated landing penalty ----
                # Scales with how far each parameter is outside the safe range.
                # A near-miss (vz=-4) gets -8; a hard crash (vz=-15) gets -35.
                # Crucially this is much smaller than hovering for many steps,
                # so the agent is never tempted to hover just to avoid landing.
                reward = (self.CRASH_BASE
                        + self.CRASH_VZ_COEF * vz_excess
                        + self.CRASH_VX_COEF * vx_excess
                        + self.CRASH_X_COEF  * x_excess)

        # Check if drone left the horizontal boundary or flew too high
        elif abs(x_new) > self.X_LIMIT or z_new > self.Z_MAX:
            terminated = True
            reward = self.CRASH_BASE - 10.0  # out-of-bounds is a bad crash

        # Check step limit — treat timeout like a bad landing (not a fixed large penalty)
        elif self.step_count >= self.MAX_STEPS:
            truncated = True
            reward = self.CRASH_BASE  # same as minimal crash: encourages landing faster

        else:
            # ---- Mid-flight rewards / penalties ----

            # Small time penalty every step
            reward += self.REWARD_TIME_STEP

            # Small penalty for thrust used (energy efficiency)
            reward += self.REWARD_THRUST_COEF * az_control

            # --- Potential-based shaping (Ng et al. 1999) ---
            # Rewards PROGRESS (improving state) not just being in a good state.
            # Hovering gives near-zero shaping; braking/steering gives positive.
            phi_old = (self.PHI_VZ * (1.0 - min(abs(vz),     self.VZ_MAX_SCALE) / self.VZ_MAX_SCALE)
                     + self.PHI_X  * (1.0 - min(abs(x),      self.X_LIMIT)      / self.X_LIMIT))
            phi_new = (self.PHI_VZ * (1.0 - min(abs(vz_new), self.VZ_MAX_SCALE) / self.VZ_MAX_SCALE)
                     + self.PHI_X  * (1.0 - min(abs(x_new),  self.X_LIMIT)      / self.X_LIMIT))
            reward += self.GAMMA_SHAPE * phi_new - phi_old

        # Update internal state
        self.state = np.array([x_new, z_new, vx_new, vz_new], dtype=np.float32)

        info = {
            "x":          x_new,
            "z":          z_new,
            "vx":         vx_new,
            "vz":         vz_new,
            "step":       self.step_count,
            "terminated": terminated,
            "truncated":  truncated,
        }

        return self.state.copy(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  render() — text-based rendering (optional, good for debugging)     #
    # ------------------------------------------------------------------ #

    def render(self):
        x, z, vx, vz = self.state
        print(f"Step {self.step_count:>4d} | "
              f"x={x:+7.2f} m  z={z:6.2f} m  "
              f"vx={vx:+6.2f} m/s  vz={vz:+6.2f} m/s")

    # ------------------------------------------------------------------ #
    #  close() — cleanup (nothing to free here yet)                       #
    # ------------------------------------------------------------------ #

    def close(self):
        pass
