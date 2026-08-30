"""
DroneEnvTau2D — tau-verrijkte drone-landingsomgeving.

Biologisch geïnspireerde uitbreiding van DroneEnv2D, gebaseerd op tau-theorie:

    τ (tau) = z / |vz|  =  geschatte time-to-contact (seconden)

Insecten (honingbijen, zweefvliegen, libellen) reguleren hun nadering zo dat
dτ/dt ongeveer constant blijft op -0.5 (Lee 1976; Wagner 1982).
Dit heet "tau-coupling" of "tau-regulatie".

    Waarom werkt dit?
    -----------------
    Als dτ/dt = κ (constant) en τ₀ = z₀/|vz₀|, dan geldt:
      - τ(t) = τ₀ + κ·t
      - Landen gebeurt wanneer τ = 0, dus op t_land = τ₀/|κ|
      - Op dat moment gaat vz ook naar 0 (omdat z → 0 en τ → 0 samen)
      - Resultaat: een ZACHTE landing — snelheid gaat naar nul zodra hoogte dat doet

    De benodigde stuwkracht voor tau-regulatie (afgeleid van d/dt van τ = z/|vz|):
      az_net = (κ + 1) · vz² / z     (voor κ = -0.5: az_net = 0.5 · vz²/z)
    Dit is altijd positief (opwaarts afremmen), wat bevestigt dat de strategie veilig is.

Wat deze omgeving toevoegt ten opzichte van DroneEnv2D:
    1. tau als 5e observatie — de agent neemt time-to-contact direct waar
    2. Tau-regulatie shaping-reward — bestraft afwijking van dτ/dt = -0.5

De landingsfysica en reward-structuur zijn IDENTIEK aan DroneEnv2D,
dus elk verschil in agentgedrag is puur toe te schrijven aan de
tau-observatie en het tau-shapingsignaal.

Gebruik:
    from envs import DroneEnvTau2D

    env = DroneEnvTau2D()
    obs, info = env.reset()
    # obs heeft vorm (5,): [x, z, vx, vz, tau]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.drone_env import DroneEnv2D


class DroneEnvTau2D(DroneEnv2D):
    """
    Breidt DroneEnv2D uit met een tau-observatie en tau-regulatie reward shaping.

    Observatieruimte (5D):
        [x, z, vx, vz, tau]
        tau = z / |vz|, geclipt naar [0, TAU_MAX]

    Extra rewardsignaal (alleen tijdens de vlucht, tijdens het dalen):
        shaping = -TAU_SHAPING_COEF * min((dτ/dt - TAU_DOT_TARGET)², MAX_ERROR_SQ)

    Alle andere rewards en stopcondities zijn overgeërfd van DroneEnv2D.
    """

    # Tau-specifieke constanten

    # Maximale tau-waarde — clipt wanneer de drone niet actief aan het dalen is.
    # 100 s is een arbitraire "heel ver weg"-waarde; VecNormalize schaalt hem toch.
    TAU_MAX: float = 100.0

    # Het biologische doel: insecten houden dτ/dt ≈ -0.5 aan
    # -1.0 zou een daling met constante snelheid betekenen (vrije val); -0.5 betekent afremmen
    TAU_DOT_TARGET: float = -0.5

    # Gewicht voor de tau-regulatie shaping-reward.
    # Bewust klein gehouden vergeleken met de tijdsstraf (-0.3/stap), zodat de
    # prikkel om veilig te landen dominant blijft; tau-shaping is een secundaire
    # biologische leidraad.
    TAU_SHAPING_COEF: float = 0.10

    # Maximale gekwadrateerde tau-fout in de straf (clipt uitschieters).
    # Zonder clipping kan één stap met een enorme tau-fout andere signalen overheersen.
    MAX_TAU_ERROR_SQ: float = 4.0

    # Tau-shaping wordt alleen toegepast tijdens een betekenisvolle daling (niet
    # vlak boven de grond waar de landingsreward het overneemt, en niet bij
    # nauwelijks beweging).
    TAU_MIN_HEIGHT:   float = 0.5    # (m) geen shaping onder deze hoogte
    TAU_MIN_VZ_MAG:   float = 0.10   # (m/s) alleen shaping bij dalen sneller dan dit

    # Constructor

    def __init__(
        self,
        render_mode=None,
        wind_force: float = 0.0,
        tau_shaping_coef: float = None,   # None = gebruik klassestandaard (0.10)
        tau_dot_target:   float = None,   # None = gebruik klassestandaard (-0.5)
    ):
        # De parent DroneEnv2D initialiseren (zet de 4D-observatieruimte,
        # actieruimte, enz. op). Geeft wind_force ook door aan de parent.
        super().__init__(render_mode, wind_force=wind_force)

        # De klasse-standaardwaarden mogen via de constructor overschreven worden.
        # Dit wordt gebruikt door train_all.py voor de sensitivity sweep.
        if tau_shaping_coef is not None:
            self.TAU_SHAPING_COEF = float(tau_shaping_coef)
        if tau_dot_target is not None:
            self.TAU_DOT_TARGET = float(tau_dot_target)

        # Observatieruimte overschrijven om de tau-dimensie toe te voegen
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

        # Tau van de vorige stap opslaan om d-tau/dt te kunnen berekenen
        self._tau_prev: float = self.TAU_MAX

    # Hulpfunctie: tau berekenen uit de huidige toestand

    def _compute_tau(self, z: float, vz: float) -> float:
        """
        tau = z / |vz|  (time-to-contact in seconden).

        Geeft TAU_MAX terug wanneer de drone niet betekenisvol aan het dalen is:
          - vz >= -TAU_MIN_VZ_MAG  (stijgen of hoveren → tau ongedefinieerd)
          - z  <= 0                (al op de grond)
        Dit voorkomt delen door nul en houdt de observatie begrensd.
        """
        if vz >= -self.TAU_MIN_VZ_MAG or z <= 0.0:
            return self.TAU_MAX
        tau = z / (-vz)   # -vz > 0 omdat vz < 0 tijdens het dalen
        return float(np.clip(tau, 0.0, self.TAU_MAX))

    # reset() — breidt de parent reset uit met tau-initialisatie

    def reset(self, *, seed=None, options=None):
        # De parent reset geeft ons een 4D-observatie en zet self.state
        obs_4d, info = super().reset(seed=seed, options=options)

        # Tau berekenen voor de startsituatie
        _, z0, _, vz0 = self.state
        tau0 = self._compute_tau(z0, vz0)
        self._tau_prev = tau0

        # Tau toevoegen aan de observatie
        obs_5d = np.append(obs_4d, tau0).astype(np.float32)

        info["tau"] = tau0
        return obs_5d, info

    # step() — breidt de parent step uit met tau-observatie en shaping

    def step(self, action: np.ndarray):
        # De natuurkunde + reward van de parent uitvoeren.
        # Na deze aanroep bevat self.state de NIEUWE (x, z, vx, vz).
        obs_4d, reward, terminated, truncated, info = super().step(action)

        # De bijgewerkte toestand uitlezen
        x_new, z_new, vx_new, vz_new = self.state

        # Nieuwe tau berekenen
        tau_new = self._compute_tau(z_new, vz_new)

        # ---- Tau-regulatie shaping-reward ----
        # Wordt alleen toegepast tijdens actief, betekenisvol dalen (niet bij
        # aanraking van de grond en niet bij nauwelijks verticale beweging).
        if (not terminated
                and not truncated
                and vz_new < -self.TAU_MIN_VZ_MAG
                and z_new  >  self.TAU_MIN_HEIGHT):

            # dτ/dt benaderen met een eindig verschil over één tijdstap
            tau_dot = (tau_new - self._tau_prev) / self.DT

            # Hoe ver zit dτ/dt van het biologische doel (-0.5) vandaan?
            tau_error = tau_dot - self.TAU_DOT_TARGET

            # Kwadratische straf, geclipt zodat één enorme stap niet gaat domineren
            tau_shaping = -self.TAU_SHAPING_COEF * min(tau_error ** 2, self.MAX_TAU_ERROR_SQ)
            reward += tau_shaping

        # tau_dot berekenen VOORDAT _tau_prev wordt bijgewerkt (gebruikt de oude waarde)
        tau_dot_info = (tau_new - self._tau_prev) / self.DT if not terminated else 0.0

        # Opgeslagen tau bijwerken voor het eindige verschil van de volgende stap
        self._tau_prev = tau_new

        # Tau toevoegen aan de info-dict zodat eval-scripts het kunnen plotten
        info["tau"]     = tau_new
        info["tau_dot"] = tau_dot_info

        # 5D-observatie teruggeven: [x, z, vx, vz, tau]
        obs_5d = np.append(obs_4d, tau_new).astype(np.float32)

        return obs_5d, reward, terminated, truncated, info

    # render() — breidt de parent uit met tau-informatie

    def render(self):
        x, z, vx, vz = self.state
        tau = self._compute_tau(z, vz)
        print(
            f"Step {self.step_count:>4d} | "
            f"x={x:+7.2f} m  z={z:6.2f} m  "
            f"vx={vx:+6.2f} m/s  vz={vz:+6.2f} m/s  "
            f"τ={tau:6.2f} s"
        )
