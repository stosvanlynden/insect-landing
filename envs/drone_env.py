"""
DroneEnv2D — een simpele 2D drone-landingsomgeving, gebouwd met Gymnasium.

Natuurkunde
-----------
De drone wordt gezien als een **puntmassa** in het verticale vlak (x, z).
Geen aerodynamica-model, alleen zwaartekracht en twee directe versnellingen:
  ax  — horizontale versnelling (besturingsinput)
  az  — netto verticale versnelling = stuwkracht - zwaartekracht

Dit is expres simpel gehouden, zodat de RL-agent zelf moet leren hoe
hij stuwkracht tegen zwaartekracht afweegt en horizontaal naar het
landingsvlak stuurt.

Coördinatensysteem
------------------
  x  — horizontale positie (m), positief naar rechts
  z  — hoogte (m), positief omhoog. De grond is z = 0.
  vx — horizontale snelheid (m/s)
  vz — verticale snelheid (m/s), negatief = dalend

Wanneer stopt een episode
--------------------------
  1. z <= 0            : drone heeft de grond geraakt (landing of crash)
  2. |x| > x_limit      : drone te ver opzij gevlogen
  3. z  > z_limit       : drone te hoog gevlogen (buiten verwacht bereik)
  4. step >= max_steps  : tijdslimiet bereikt

Reward-functie (ontworpen voor een veilige landing rond x=0, z=0)
------------------------------------------------------------------
  Veilige landing : grote positieve reward, geschaald op hoe dicht bij
                     het doel en hoe zacht de landing is
  Crash            : negatieve straf
  Stuwkracht-straf : kleine straf per stap, proportioneel aan |az_control|
                      (stimuleert energiezuinigheid — insect-geïnspireerde spaarzaamheid)
  Tijdsstraf       : klein negatief signaal elke stap (stimuleert efficiëntie)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DroneEnv2D(gym.Env):
    # Constanten op klasse-niveau — natuurkundige en episode-parameters

    # Valversnelling (m/s²)
    GRAVITY: float = 9.81

    # Simulatie-tijdstap (s) — update-frequentie van 10 Hz
    DT: float = 0.1

    # Maximale episodelengte (stappen)
    MAX_STEPS: int = 500

    # Ruimtelijke grenzen — episode stopt als de drone deze box verlaat
    X_LIMIT: float = 30.0   # (m) horizontaal
    Z_MAX:   float = 50.0   # (m) maximale hoogte

    # Actiegrenzen (m/s²)
    MAX_AX: float = 5.0     # max. horizontale versnelling
    MAX_AZ: float = 20.0    # max. opwaartse stuwkracht-versnelling
                             # netto omhoog = 20 - 9.81 = 10.19 m/s² — genoeg om
                             # binnen ~1 seconde af te remmen vanaf -10 m/s
                             # let op: netto verticale versnelling = az_thrust - GRAVITY

    # Landingsdrempels voor een geslaagde landing
    LAND_X_THRESHOLD:  float = 2.0   # (m)   moet binnen dit bereik van x=0 landen
    LAND_VX_THRESHOLD: float = 2.0   # (m/s) max. horizontale snelheid bij aanraking
    LAND_VZ_THRESHOLD: float = 3.0   # (m/s) max. neerwaartse snelheid bij aanraking
                                      #        vz is negatief tijdens dalen

    # Reward-gewichten
    REWARD_SAFE_BASE:   float =  100.0  # max. reward voor een perfecte veilige landing
    REWARD_TIME_STEP:   float =   -0.3  # kleine straf elke stap (stimuleert snelheid)
    REWARD_THRUST_COEF: float =   -0.01 # kleine straf per (m/s²) gebruikte stuwkracht

    # --- Graduele landingsstraf ---
    # In plaats van een vaste crash-straf van -100, schaalt de straf mee met
    # HOE SLECHT de landing is. Dit is de kernfix voor het hover-probleem:
    #
    #   Oud ontwerp: crash = -100, 500 stappen hoveren = -250 → agent kiest liever crash
    #                maar crash = -100 is nog steeds erger dan 100 stappen hoveren (-50)
    #                → agent leert 100 stappen te hoveren en dan een crash te accepteren.
    #
    #   Nieuw ontwerp: slechte landing = -5 tot -40 (veel kleiner dan 500 stappen
    #                  hoveren = -150)
    #                → SNEL crashen is altijd beter dan hoveren
    #                → een veilige landing (= +100) is veel beter dan elke crash
    #                → agent wordt richting veilig landen gestuurd, niet richting hoveren
    #
    # Strafformule bij landen:
    #   basis            : -5.0  (altijd, voor niet-veilig landen)
    #   snelheidsoverschot: -3.0 * max(0, |vz| - VZ_THRESHOLD)   per m/s boven de limiet
    #   zijwaarts overschot: -2.0 * max(0, |vx| - VX_THRESHOLD)
    #   positie-overschot : -1.0 * max(0, |x|  - X_THRESHOLD)
    CRASH_BASE:        float = -5.0
    CRASH_VZ_COEF:     float = -3.0
    CRASH_VX_COEF:     float = -2.0
    CRASH_X_COEF:      float = -1.0

    # --- Potential-based reward shaping (Ng et al. 1999) ---
    # F = gamma * Phi(s') - Phi(s), toegevoegd aan elke tussenstap.
    # Geeft een gradiënt richting lagere |vz| en lagere |x| ZONDER hoveren te belonen.
    GAMMA_SHAPE:  float = 0.99
    PHI_VZ:       float = 2.0    # gewicht van het verticale-snelheid-potentiaal
    PHI_X:        float = 0.5    # gewicht van het horizontale-positie-potentiaal
    VZ_MAX_SCALE: float = 20.0

    # Constructor

    def __init__(self, render_mode=None, wind_force: float = 0.0):
        super().__init__()

        self.render_mode = render_mode  # placeholder (nog geen live rendering)

        # Optionele windverstoring (m/s²).
        # Elke stap wordt een willekeurige horizontale versnelling in
        # [-wind_force, +wind_force] bij ax opgeteld. 0.0 = geen wind
        # (standaard, gebruikt tijdens training).
        # Alleen aanzetten tijdens EVALUATIE, om zero-shot robuustheid te testen.
        self.wind_force: float = wind_force

        # --- Actieruimte ---
        # Twee continue acties: [ax, az_control]
        #   ax          in [-MAX_AX,  +MAX_AX]   horizontale versnelling
        #   az_control  in [0,        +MAX_AZ]   opwaartse stuwkracht-versnelling
        #               (kan niet naar beneden duwen; dat doet de zwaartekracht al)
        self.action_space = spaces.Box(
            low  = np.array([-self.MAX_AX, 0.0],        dtype=np.float32),
            high = np.array([ self.MAX_AX, self.MAX_AZ], dtype=np.float32),
        )

        # --- Observatieruimte ---
        # Vier continue toestandsvariabelen: [x, z, vx, vz]
        obs_low  = np.array([-self.X_LIMIT, 0.0,        -20.0, -20.0], dtype=np.float32)
        obs_high = np.array([ self.X_LIMIT, self.Z_MAX,  20.0,  20.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Interne toestand (wordt goed gezet in reset())
        self.state: np.ndarray = np.zeros(4, dtype=np.float32)
        self.step_count: int = 0

    # reset() — wordt aangeroepen bij het begin van elke nieuwe episode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # zet self.np_random via Gymnasium

        # Startpositie willekeurig kiezen binnen een redelijke box.
        # Hoogtes blijven wat lager (3-12 m) zodat de drone genoeg hoogte
        # heeft om af te remmen met de beschikbare stuwkracht voordat hij
        # de grond raakt. Beginsnelheden starten dicht bij nul, zodat de
        # agent eerst leert hoveren en sturen, en daarna generaliseert
        # naar lastigere startsituaties.
        x0  = self.np_random.uniform(-8.0,  8.0)   # horizontale afwijking (m)
        z0  = self.np_random.uniform( 3.0, 12.0)   # starthoogte (m)
        vx0 = self.np_random.uniform(-0.5,  0.5)   # bijna geen horizontale drift
        vz0 = self.np_random.uniform(-0.5,  0.2)   # bijna geen verticale snelheid

        self.state = np.array([x0, z0, vx0, vz0], dtype=np.float32)
        self.step_count = 0

        info = {}
        return self.state.copy(), info

    # step() — simulatie één tijdstap verder zetten

    def step(self, action: np.ndarray):
        # Huidige toestand uitpakken
        x, z, vx, vz = self.state

        # Acties clippen naar geldig bereik (voor de zekerheid — voor het geval
        # de agent een NaN produceert)
        ax          = float(np.clip(action[0], -self.MAX_AX, self.MAX_AX))
        az_control  = float(np.clip(action[1],  0.0,          self.MAX_AZ))

        # Windverstoring toevoegen aan de horizontale versnelling.
        # np_random wordt geseed door reset(), dus wind is reproduceerbaar per seed.
        if self.wind_force > 0.0:
            wind = self.np_random.uniform(-self.wind_force, self.wind_force)
            ax = float(np.clip(ax + wind, -self.MAX_AX * 2, self.MAX_AX * 2))

        # Netto verticale versnelling: stuwkracht omhoog min zwaartekracht omlaag
        az_net = az_control - self.GRAVITY

        # ------ Euler-integratie (simpel maar doorzichtig) ------
        # Eerst de snelheden updaten
        vx_new = vx + ax         * self.DT
        vz_new = vz + az_net     * self.DT

        # Posities updaten met de nieuwe snelheden (semi-impliciete Euler)
        x_new  = x  + vx_new    * self.DT
        z_new  = z  + vz_new    * self.DT

        self.step_count += 1

        # ------ Bepalen of de episode stopt ------
        terminated = False   # natuurlijk einde (grondcontact, grens)
        truncated  = False   # kunstmatig einde (staplimiet)
        reward     = 0.0

        # Check of de drone de grond heeft geraakt (z <= 0)
        if z_new <= 0.0:
            terminated = True
            z_new = 0.0  # vastzetten op grondniveau

            # Hoeveel overschrijdt elke dimensie de veilige landingsdrempel?
            vz_excess  = max(0.0, abs(vz_new) - self.LAND_VZ_THRESHOLD)
            vx_excess  = max(0.0, abs(vx_new) - self.LAND_VX_THRESHOLD)
            x_excess   = max(0.0, abs(x_new)  - self.LAND_X_THRESHOLD)

            if vz_excess == 0.0 and vx_excess == 0.0 and x_excess == 0.0:
                # ---- Veilige landing ----
                # Reward schaalt mee met hoe gecentreerd en zacht de landing is.
                position_score = 1.0 - abs(x_new)  / self.LAND_X_THRESHOLD
                speed_score    = 1.0 - abs(vz_new) / self.LAND_VZ_THRESHOLD
                reward = self.REWARD_SAFE_BASE * (0.5 * position_score + 0.5 * speed_score)
            else:
                # ---- Graduele landingsstraf ----
                # Schaalt mee met hoe ver elke variabele buiten het veilige bereik zit.
                # Een bijna-misser (vz=-4) krijgt -8; een harde crash (vz=-15) krijgt -35.
                # Dit is cruciaal veel kleiner dan lang hoveren, zodat de agent
                # nooit de verleiding voelt om te hoveren om landen te vermijden.
                reward = (self.CRASH_BASE
                        + self.CRASH_VZ_COEF * vz_excess
                        + self.CRASH_VX_COEF * vx_excess
                        + self.CRASH_X_COEF  * x_excess)

        # Check of de drone buiten de horizontale grens is gevlogen of te hoog
        elif abs(x_new) > self.X_LIMIT or z_new > self.Z_MAX:
            terminated = True
            reward = self.CRASH_BASE - 10.0  # buiten de grenzen is een zware crash

        # Check staplimiet — behandel een timeout als een slechte landing
        # (geen vaste grote straf)
        elif self.step_count >= self.MAX_STEPS:
            truncated = True
            reward = self.CRASH_BASE  # zelfde als een minimale crash: stimuleert sneller landen

        else:
            # ---- Rewards/straffen tijdens de vlucht ----

            # Kleine tijdsstraf elke stap
            reward += self.REWARD_TIME_STEP

            # Kleine straf voor gebruikte stuwkracht (energiezuinigheid)
            reward += self.REWARD_THRUST_COEF * az_control

            # --- Potential-based shaping (Ng et al. 1999) ---
            # Beloont VOORUITGANG (een betere toestand) en niet zomaar het
            # ín een goede toestand zijn. Hoveren geeft bijna geen shaping;
            # afremmen/sturen geeft een positieve waarde.
            phi_old = (self.PHI_VZ * (1.0 - min(abs(vz),     self.VZ_MAX_SCALE) / self.VZ_MAX_SCALE)
                     + self.PHI_X  * (1.0 - min(abs(x),      self.X_LIMIT)      / self.X_LIMIT))
            phi_new = (self.PHI_VZ * (1.0 - min(abs(vz_new), self.VZ_MAX_SCALE) / self.VZ_MAX_SCALE)
                     + self.PHI_X  * (1.0 - min(abs(x_new),  self.X_LIMIT)      / self.X_LIMIT))
            reward += self.GAMMA_SHAPE * phi_new - phi_old

        # Interne toestand bijwerken
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

    # render() — tekstweergave (optioneel, handig om te debuggen)

    def render(self):
        x, z, vx, vz = self.state
        print(f"Step {self.step_count:>4d} | "
              f"x={x:+7.2f} m  z={z:6.2f} m  "
              f"vx={vx:+6.2f} m/s  vz={vz:+6.2f} m/s")

    # close() — opruimen (hier nog niets om vrij te geven)

    def close(self):
        pass
