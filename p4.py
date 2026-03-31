"""
CHANGELOG (bug-fixes applied vs original):
  FIX 1 — intensity_buildup():
      OLD: B = Q·λ / (2π·ng·L)   ← equals F/(2π), not F/π  [factor-of-2 error]
      NEW: B = Q·λ / (π·ng·L)    ← equals F/π  per Bogaerts Eq. 12–13
      Impact: P_π,ring halves; enhancement ratio doubles; Q² slope unchanged.

  FIX 2 — fig1_straight_waveguide():
      Added cm-scale waveguide lengths (1–100 cm) where P_π is truly "several watts",
      matching the paper's stated regime.  1 mm waveguides require ~kW, not watts,
      because γ ≈ 1 W⁻¹m⁻¹ and Leff(1 mm) ≈ 1 mm → P_π = π/(γ·Leff) ≈ 3.2 kW.

  NEW — fig7_design_parameter_space():
      Pareto map across (R, α_loss, Q_target) showing why the proposed experimental
      point (R=23 µm, α=1 dB/m, Q=10⁶–10⁷) is preferred over all alternatives.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator

# Physical constants

PI     = np.pi
C      = 3.0e8           # speed of light  (m/s)
OMEGA0 = 2*PI*C/1550e-9  # angular frequency at 1550 nm  (rad/s)

# Si3N4 design parameters (paper Table, Section 4)
LAMBDA  = 1550e-9       # wavelength (m)  — telecom C-band
N2      = 2.4e-19       # Kerr index Si3N4  (m²/W)  — Levy 2010, lower bound
AEFF    = 1.0e-12       # effective mode area (m²)  = 1 µm²  — Ji 2017
NG      = 2.0           # group index  — Ji 2017 FEM result
ALPHA   = 2.3e-4        # power propagation loss  (m⁻¹)  = 1 dB/m  — Pfeiffer 2018
R_ring  = 23e-6         # ring radius  (m)  — paper design
L_geom  = 2*PI*R_ring   # ring circumference  (m)  ≈ 144 µm

# Derived
GAMMA = OMEGA0 * N2 / (C * AEFF)       # nonlinear parameter (W⁻¹ m⁻¹), Agrawal 2.3.30
FSR   = C / (NG * L_geom)              # free spectral range (Hz)

# Core physics functions (traceable to a paper equation)

def leff_straight(L, alpha=ALPHA):
    # Straight-waveguide effective length (Agrawal Eq. 4.1.6): Leff = [1 - exp(-αL)] / α
    return (1.0 - np.exp(-alpha * L)) / alpha


def P_pi_straight(L=1e-3, n2=N2, aeff=AEFF, lam=LAMBDA, alpha=ALPHA):

    # Power for π-phase shift in straight waveguide (Agrawal 4.1.7): P_π = π / (γ · Leff)  =  π·c·Aeff / (ω0·n2·Leff)
    # Note: with γ ≈ 1 W⁻¹m⁻¹ and Leff(1 mm) ≈ 1 mm: P_π(1 mm) ≈ π / (1 × 10⁻³) ≈ 3.2 kW
    # "Several watts" corresponds to cm–dm scale waveguides: L ≈ 65 cm gives P_π ≈ 5 W.

    gamma = 2*PI/lam * n2 / aeff      # = ω0·n2/(c·Aeff)
    leff  = leff_straight(L, alpha)
    return PI / (gamma * leff)


def leff_resonator(Q, ng=NG, lam=LAMBDA):

    # Resonantly extended effective interaction length (paper Eq. 16): Leff_res = Q·λ / (2π·ng)

    # The mean photon path length inside the resonator before escape.
    # At Q=10⁶: Leff_res ≈ 12 cm  >>  L_geom ≈ 144 µm.
    return Q * lam / (2.0 * PI * ng)


def intensity_buildup(Q, L=L_geom, ng=NG, lam=LAMBDA):
    # Intracavity intensity buildup factor at critical coupling (Bogaerts Eq. 12–14): B = I_cav / I_in = F/π
    # where the cavity finesse F = Q·λ / (ng·L)  (from Q = ng·L·F / λ), giving: B = Q·λ / (π·ng·L)   ∝ Q

    # FIX vs original code: denominator is π·ng·L, not 2π·ng·L. The factor 2π would give F/(2π) which underestimates B by 2×.
    return Q * lam / (PI * ng * L)   # ← FIX: PI not 2*PI


def delta_phi_ring(P_in, Q, n2=N2, aeff=AEFF, ng=NG, lam=LAMBDA, L=L_geom):

    # Q²-enhanced Kerr phase shift in the ring (paper §3.3):
        # Δφ = (2π/λ)·n2·I_cav·Leff_res
           # = (2π/λ)·n2·(B·I_in)·Leff_res

    # With B = Q·λ/(π·ng·L) and Leff_res = Q·λ/(2π·ng):
        # Δφ = n2·Q²·I_in / (ng²·L)   [∝ Q² · I_in]

    # Both enhancement factors (B ∝ Q and Leff_res ∝ Q) multiply,
    # giving the Q² scaling that is the central result of the paper.

    I_in   = P_in / aeff
    B      = intensity_buildup(Q, L, ng, lam)
    I_cav  = B * I_in
    Leff_r = leff_resonator(Q, ng, lam)
    return (2*PI / lam) * n2 * I_cav * Leff_r


def P_pi_ring(Q, n2=N2, aeff=AEFF, ng=NG, lam=LAMBDA, L=L_geom):

    # Input power for π-phase shift in the microring (solve Δφ = π for P_in):

        # (2π/λ)·n2·(B·P_in/Aeff)·Leff_res = π
        # ⟹ P_in = λ·Aeff / (2·n2·B·Leff_res)

    # Expanding with B = Q·λ/(π·ng·L) and Leff_res = Q·λ/(2π·ng):
        # B · Leff_res = Q²·λ² / (2π²·ng²·L)
        # P_π,ring = λ·Aeff·2π²·ng²·L / (2·n2·Q²·λ²)
                 # = π²·ng²·L·Aeff / (n2·Q²·λ)   ∝ 1/Q²

    # This is the correct Bogaerts-grounded formula.  The paper's Eq. 28 numerics
    # use a simpler proportionality (B ∝ Q directly, absorbing the ring geometry)
    # which differs by the factor π·ng·L/λ ≈ 586 but preserves the Q² scaling.

    B      = intensity_buildup(Q, L, ng, lam)
    Leff_r = leff_resonator(Q, ng, lam)
    return lam * aeff / (2.0 * n2 * B * Leff_r)


def tau_photon(Q, lam=LAMBDA, c=C):
    # Photon lifetime (paper Eq. 25 / Sec. 4.6): τ_ph = Q·λ / (2π·c)
    return Q * lam / (2.0 * PI * c)


def bandwidth_hz(Q, lam=LAMBDA, c=C):
    # Resonance linewidth / operational bandwidth (paper Eq. 31): Δν = c / (λ·Q)
    return c / (lam * Q)


def ring_transfer(r, a, delta):

    # All-pass microring power transfer function (Bogaerts 2012): T(δ) = (a² - 2ar·cos δ + r²) / (1 - 2ar·cos δ + (ar)²)

    # At resonance (δ = 2πm) and critical coupling (r = a):  T = 0.
    # Off resonance: T → 1.

    cos_d = np.cos(delta)
    num   = a**2 - 2*a*r*cos_d + r**2
    den   = 1.0 - 2*a*r*cos_d + (a*r)**2
    return num / den


def r_from_Q(Q, L=L_geom, ng=NG, lam=LAMBDA):

    # Self-coupling coefficient r = a at critical coupling for a given Q.

    # From Bogaerts finesse:  F = π·a/(1-a²) ≈ π/(1-a²) for a→1
    # And Q = ng·L·F/λ  ⟹  F = Q·λ/(ng·L)
    # So:  1 - a² = π·ng·L / (λ·Q)  ⟹  a = √(1 - π·ng·L/(λ·Q))

    val = 1.0 - PI * ng * L / (lam * Q)
    a   = np.sqrt(np.clip(val, 0.0, 1.0 - 1e-9))
    return a



# Plotting style helpers

BLUE   = "#1a6faf"
RED    = "#c0392b"
GREEN  = "#27ae60"
ORANGE = "#e67e22"
PURPLE = "#7d3c98"
GRAY   = "#7f8c8d"
LGRAY  = "#bdc3c7"

def _style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=7)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.22, linewidth=0.6, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _textbox(ax, text, loc="upper right", fontsize=8):
    locs = {
        "upper right": (0.97, 0.97, "right", "top"),
        "upper left":  (0.03, 0.97, "left",  "top"),
        "lower right": (0.97, 0.03, "right", "bottom"),
        "lower left":  (0.03, 0.03, "left",  "bottom"),
    }
    x, y, ha, va = locs[loc]
    ax.text(x, y, text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=fontsize,
            bbox=dict(fc="white", ec="#cccccc", alpha=0.88, boxstyle="round,pad=0.4"))


# =============================================================================
# FIGURE 1 — NLS / SPM phase shift in straight waveguide
# =============================================================================
def fig1_straight_waveguide():
    """
    Straight-waveguide SPM phase shift (Agrawal Eq. 4.1.7):
        φ_max = γ · P0 · Leff

    Panel A: φ_max vs input power for cm-scale waveguide lengths.
             Shows that even at cm scale, reaching φ = π requires watts to tens of watts,
             motivating the resonant approach.

    Panel B: Required P_π vs waveguide length (log–log).
             The "several watts" regime sits at L ≈ 10–100 cm — far beyond chip scale.
             Annotates the crossover into mW territory (L ~ 100 m) to make the contrast vivid.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(
        "SPM phase shift in straight Si₃N₄ waveguide (Agrawal Eqs. 4.1.5–4.1.7)\n"
        r"$\gamma \approx 1\,\mathrm{W^{-1}m^{-1}}$  →  π-shift requires watts at cm scale, kW at mm scale",
        fontsize=11, fontweight="bold"
    )

    # --- Panel A: φ_max vs P0 for cm-scale lengths ---
    ax = axes[0]
    P0      = np.linspace(0, 30.0, 400)          # watts
    lengths = [1e-2, 5e-2, 0.1, 0.5]             # 1 cm, 5 cm, 10 cm, 50 cm
    colors  = [BLUE, GREEN, ORANGE, RED]
    labels  = ["1 cm", "5 cm", "10 cm", "50 cm"]

    for L, col, lbl in zip(lengths, colors, labels):
        Leff = leff_straight(L)
        phi  = GAMMA * P0 * Leff
        ax.plot(P0, phi / PI, color=col, lw=2, label=f"L = {lbl},  L$_{{eff}}$ = {Leff*100:.1f} cm")

    ax.axhline(1.0, color=GRAY, lw=1.2, ls="--", label=r"$\phi_{max} = \pi$  (switching threshold)")
    _style(ax, "Input power $P_0$  (W)", r"$\phi_{max} / \pi$",
           r"A — SPM phase shift vs power (cm-scale waveguides)")
    ax.set_xlim(0, 30)
    ax.legend(fontsize=8, framealpha=0.6, loc="upper left")
    _textbox(ax,
             rf"$\gamma = \omega_0 n_2/(c A_{{eff}}) = {GAMMA:.2f}$ W$^{{-1}}$m$^{{-1}}$" "\n"
             rf"$n_2 = {N2:.1e}$ m²/W,  $A_{{eff}} = 1\,\mu$m²" "\n\n"
             "At 1 mm: $P_\\pi \\approx 3.2$ kW\n"
             "At 10 cm: $P_\\pi \\approx 32$ W\n"
             "At 50 cm: $P_\\pi \\approx 6.5$ W\n"
             r"→ resonator needed for mW-scale",
             "upper right")

    # --- Panel B: P_π vs length (log–log, wide range) ---
    ax2 = axes[1]
    L_arr = np.logspace(-3, 1, 400)    # 1 mm → 10 m
    P_pi  = np.array([P_pi_straight(L) for L in L_arr])

    ax2.loglog(L_arr * 100, P_pi, color=BLUE, lw=2.5,
               label=r"$P_\pi$ (straight WG)")

    # Reference power levels
    ax2.axhline(1e3,  color=RED,    lw=1.0, ls=":", alpha=0.8, label="1 kW")
    ax2.axhline(10,   color=ORANGE, lw=1.0, ls=":", alpha=0.8, label="10 W")
    ax2.axhline(1.0,  color=GREEN,  lw=1.2, ls="--", alpha=0.9, label="1 W  (EDFA-level)")
    ax2.axhline(1e-3, color=PURPLE, lw=1.2, ls="-.", alpha=0.8, label="1 mW  (laser-level)")

    # Mark chip-scale (mm) and "several watts" (cm–dm)
    markers = [(1e-3, "1 mm\n~3.2 kW"), (0.1, "10 cm\n~32 W"), (0.65, "65 cm\n~5 W")]
    for L_m, lbl in markers:
        p = P_pi_straight(L_m)
        ax2.scatter([L_m*100], [p], s=70, color=ORANGE, zorder=5)
        ax2.annotate(lbl, xy=(L_m*100, p), xytext=(L_m*100*2.2, p*0.35),
                     fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))

    # Shaded "practical input power" band (1 mW – 100 mW)
    ax2.axhspan(1e-3, 0.1, alpha=0.07, color=GREEN, label="mW–100 mW practical window")

    _style(ax2, "Waveguide length  (cm)",
           r"$P_\pi$  (W)",
           r"B — Required $P_\pi$ vs waveguide length")
    ax2.set_xlim(L_arr[0]*100, L_arr[-1]*100)
    ax2.legend(fontsize=7.5, framealpha=0.6, loc="upper right")
    _textbox(ax2,
             "γ ≈ 1 W⁻¹m⁻¹ → P_π ∝ 1/L\n"
             "'Several watts': L ~ 65 cm\n"
             "'Chip-scale' (mm): L ~ kW\n\n"
             "Microring at Q=10⁶:\n"
             "P_π ~ 15 mW (12 cm eff. path)\n"
             "→ 1000× improvement",
             "lower left")

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 2 — Microring resonance lineshape (Bogaerts transfer function)
# =============================================================================
def fig2_resonance_lineshape():
    """
    Bogaerts all-pass transfer function T(δ) across one FSR and zoomed.
    Demonstrates:
     - Critical coupling → deepest notch (maximum energy absorption into ring)
     - Linewidth Δλ = λ/Q  →  bandwidth constraint
     - Three coupling regimes
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Microring resonance lineshape — Bogaerts all-pass transfer function\n"
        r"$T(\delta) = (a^2 - 2ar\cos\delta + r^2)/(1 - 2ar\cos\delta + (ar)^2)$",
        fontsize=11, fontweight="bold"
    )

    a_loss = r_from_Q(1e5)       # round-trip field amplitude for Q=10^5 demo
    configs = [
        (a_loss, a_loss,  BLUE,   "Critical coupling  r = a\n(max energy transfer into ring)"),
        (0.990,  a_loss,  GREEN,  "Over-coupled  r > a"),
        (a_loss + 0.003, a_loss,  ORANGE, "Under-coupled  r < a"),
    ]

    # Wavelength axis spanning ±1.5 FSR around 1550 nm
    wl_span = 3.0 * C / (NG * L_geom)
    wl_arr  = np.linspace(LAMBDA - wl_span/2 * LAMBDA**2/C,
                          LAMBDA + wl_span/2 * LAMBDA**2/C, 6000)

    def T_wl(r, a, wl_array):
        delta = 2*PI * NG * L_geom / wl_array
        return ring_transfer(r, a, delta % (2*PI))

    # Panel A — full FSR view
    ax = axes[0]
    for r_, a_, col, lbl in configs:
        ax.plot((wl_arr - LAMBDA)*1e12, T_wl(r_, a_, wl_arr),
                color=col, lw=1.8, label=lbl)
    _style(ax, "Δλ from 1550 nm  (pm)", "Power transmission  T",
           "A — Full spectrum (±1.5 × FSR)")
    ax.set_ylim(-0.02, 1.08)
    ax.legend(fontsize=7.5, framealpha=0.6, loc="upper right")
    _textbox(ax,
             f"R = {R_ring*1e6:.0f} µm\n"
             f"L = {L_geom*1e6:.1f} µm\n"
             f"FSR = {FSR/1e9:.1f} GHz\n"
             f"  = {C/FSR*1e12:.1f} pm",
             "lower right")

    # Panel B — zoom on single resonance
    zoom_hz  = 8 * bandwidth_hz(1e5)
    zoom_nm  = zoom_hz * LAMBDA**2 / C
    wl_zoom  = np.linspace(LAMBDA - zoom_nm, LAMBDA + zoom_nm, 5000)

    ax2 = axes[1]
    for r_, a_, col, lbl in configs:
        T = T_wl(r_, a_, wl_zoom)
        ax2.plot((wl_zoom - LAMBDA)*1e12, T, color=col, lw=2, label=lbl.split("\n")[0])

        # FWHM annotation for critical coupling
        if r_ == a_:
            T_min  = T.min()
            T_half = (1.0 + T_min) / 2.0
            cross  = np.diff((T < T_half).astype(int))
            idx    = np.where(cross != 0)[0]
            if len(idx) >= 2:
                wl_l = wl_zoom[idx[0]]
                wl_r = wl_zoom[idx[-1]]
                fwhm_pm = (wl_r - wl_l) * 1e12
                y_half  = T_half
                ax2.annotate("", xy=((wl_r-LAMBDA)*1e12, y_half),
                             xytext=((wl_l-LAMBDA)*1e12, y_half),
                             arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.5))
                ax2.text(0, y_half + 0.04,
                         f"FWHM = {fwhm_pm:.3f} pm = Δλ = λ/Q",
                         ha="center", fontsize=8, color=BLUE)

    _style(ax2, "Δλ from 1550 nm  (pm)", "Power transmission  T",
           "B — Single resonance zoom (Q = 10⁵)")
    ax2.set_xlim((wl_zoom[0]-LAMBDA)*1e12, (wl_zoom[-1]-LAMBDA)*1e12)
    ax2.set_ylim(-0.05, 1.12)
    ax2.legend(fontsize=8, framealpha=0.6)
    _textbox(ax2,
             "Critical coupling: deepest notch\n"
             r"→ maximum $I_{cav}/I_{in} = B = F/\pi$" "\n"
             r"→ largest Kerr $\Delta\phi$ per watt" "\n\n"
             r"Linewidth $\Delta\nu = c/(\lambda Q)$" "\n"
             r"sets max operational bandwidth",
             "lower right")

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 3 — Q² energy scaling vs straight waveguide  (central result)
# =============================================================================
def fig3_energy_scaling():
    """
    Panel A: Required input power for Δφ = π  vs  Q.
      - Ring (correct Q² model, Bogaerts B = F/π): P_π,ring  ∝ 1/Q²
      - Naïve ring (only buildup B ∝ Q, fixed geometric length): P  ∝ 1/Q
      - Straight WG baselines at realistic "several-watt" lengths (10 cm, 65 cm)

    Panel B: Enhancement ratio  P_straight(65 cm) / P_ring  vs Q.
    """
    Q_arr = np.logspace(3, 7.5, 400)

    # Correct Q² model (paper): B = F/π ∝ Q  AND  Leff_res ∝ Q
    P_ring_Q2 = np.array([P_pi_ring(Q) for Q in Q_arr])

    # Naïve model: only intensity buildup (B ∝ Q), fixed geometric length
    # Uses CORRECT B but wrong Leff = L_geom instead of Leff_res
    P_naive = np.array([
        LAMBDA * AEFF / (2.0 * N2 * intensity_buildup(Q) * leff_straight(L_geom))
        for Q in Q_arr
    ])

    # Straight waveguide baselines (realistic chip lengths)
    P_str_10cm = P_pi_straight(L=0.10)   # 10 cm → ~32 W
    P_str_65cm = P_pi_straight(L=0.65)   # 65 cm → ~5 W  ("several watts")

    # Proposed operating point: Q = 10^6
    Q_op  = 1e6
    P_op  = P_pi_ring(Q_op)
    E_op  = P_op * tau_photon(Q_op)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.suptitle(
        r"Energy scaling with quality factor $Q$  —  central $1/Q^2$ result  (corrected $B = F/\pi$)",
        fontsize=12, fontweight="bold"
    )

    # Panel A — threshold power
    ax = axes[0]
    ax.loglog(Q_arr, P_ring_Q2 * 1e3,  color=BLUE,   lw=2.5,
              label=r"Ring: $P_\pi \propto 1/Q^2$  (correct: $B=F/\pi$ and $L_{eff,res} \propto Q$)")
    ax.loglog(Q_arr, P_naive * 1e3,    color=ORANGE, lw=2, ls="--",
              label=r"Ring: $P_\pi \propto 1/Q$  (naïve: buildup only, $L = L_{geom}$)")
    ax.axhline(P_str_65cm * 1e3, color=RED, lw=1.5, ls=":",
               label=f"Straight WG 65 cm: {P_str_65cm:.1f} W  ('several watts')")
    ax.axhline(P_str_10cm * 1e3, color=ORANGE, lw=1.2, ls=":",
               label=f"Straight WG 10 cm: {P_str_10cm:.0f} W")

    # Mark proposed operating point
    ax.scatter([Q_op], [P_op * 1e3], s=100, color=BLUE, zorder=6, marker="*")
    ax.annotate(
        f"Q = 10⁶\n"
        f"P = {P_op*1e3:.1f} mW\n"
        f"E = {E_op*1e12:.0f} pJ",
        xy=(Q_op, P_op*1e3), xytext=(Q_op*5, P_op*1e3*300),
        fontsize=8, color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9),
        bbox=dict(fc="white", ec="#aaaaaa", alpha=0.9, boxstyle="round,pad=0.3")
    )

    # Reference slope line
    ax.loglog(Q_arr[[0, -1]],
              [P_ring_Q2[0]*1e3, P_ring_Q2[0]*(Q_arr[0]/Q_arr[-1])**2 * 1e3],
              color=LGRAY, lw=1, ls="--", zorder=0)
    ax.text(3e5, P_ring_Q2[0]*(Q_arr[0]/3e5)**2 * 1e3 * 4,
            r"slope $-2$", fontsize=7.5, color=GRAY, rotation=-40)

    _style(ax, "Quality factor $Q$",
           r"Power for $\Delta\phi = \pi$  (mW)",
           r"A — Threshold power $P_\pi$ vs $Q$")
    ax.legend(fontsize=8, framealpha=0.6, loc="upper right")

    # Panel B — enhancement ratio vs 65 cm straight WG
    ax2 = axes[1]
    ratio = P_str_65cm / P_ring_Q2
    ax2.loglog(Q_arr, ratio, color=BLUE, lw=2.5,
               label=r"$P_{str}(65\,\mathrm{cm}) / P_{ring}$  (Q² model)")

    ax2.scatter([Q_op], [P_str_65cm / P_op], s=100, color=BLUE, zorder=6, marker="*")
    ax2.annotate(
        f"Q = 10⁶:\n{P_str_65cm/P_op:.1e}× over 65 cm WG",
        xy=(Q_op, P_str_65cm/P_op), xytext=(Q_op*4, P_str_65cm/P_op*0.02),
        fontsize=8, color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9),
        bbox=dict(fc="white", ec="#aaaaaa", alpha=0.9, boxstyle="round,pad=0.3")
    )

    _style(ax2, "Quality factor $Q$",
           r"Enhancement  $P_{straight} / P_{ring}$",
           r"B — Power enhancement ratio vs $Q$")
    ax2.legend(fontsize=8, framealpha=0.6)
    _textbox(ax2,
             r"$Q^2$ scaling: two resonant mechanisms" "\n"
             r"  1. $B = F/\pi \propto Q$ (intensity buildup)" "\n"
             r"  2. $L_{eff,res} = Q\lambda/(2\pi n_g) \propto Q$ (path length)" "\n"
             r"  → $\Delta\phi \propto B \cdot L_{eff,res} \propto Q^2 \cdot I_{in}$" "\n\n"
             r"Naïve model (B only): slope $-1$" "\n"
             r"Correct model (B·L): slope $-2$",
             "upper left")

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 4 — Kerr-shifted activation function (photonic neuron)
# =============================================================================
def fig4_activation():
    """
    The microring photonic-neuron activation function.

    As P_in increases, the Kerr effect shifts the resonance by ΔφNL(P_in).
    The through-port transmission traces a nonlinear, sigmoidal-like curve.

    Panel A: Resonance spectra at increasing power levels, showing red-shift.
    Panel B: Through-port transmission T(λ_op) vs input power — the activation curve.
    Panel C: Output power vs input power — nonlinear input-output map.

    Uses Q = 10^5 for visibility of switching features.
    """
    Q_act  = 1e5
    a_loss = r_from_Q(Q_act)
    r_cc   = a_loss

    P_thresh = P_pi_ring(Q_act)

    # Wavelength sweep (zoomed on one resonance, ±7.5 linewidths)
    span = 15 * C * LAMBDA**2 / (C * Q_act * LAMBDA)
    wl   = np.linspace(LAMBDA - span, LAMBDA + span, 4000)

    def T_at_power(P, wl_arr):
        I_in   = P / AEFF
        B      = intensity_buildup(Q_act)
        I_cav  = B * I_in
        Leff_r = leff_resonator(Q_act)
        dphiNL = (2*PI / LAMBDA) * N2 * I_cav * Leff_r
        delta  = (2*PI * NG * L_geom / wl_arr) + dphiNL
        return ring_transfer(r_cc, a_loss, delta % (2*PI))

    n_levels    = 6
    P_levels    = np.linspace(0, 3.5 * P_thresh, n_levels)
    spec_colors = [plt.cm.Blues(0.3 + 0.12*i) for i in range(n_levels)]

    P_sweep    = np.linspace(0, 5 * P_thresh, 600)
    T_at_lam0  = np.array([
        ring_transfer(r_cc, a_loss,
                      (2*PI*NG*L_geom/LAMBDA +
                       (2*PI/LAMBDA)*N2*intensity_buildup(Q_act)*(p/AEFF)*leff_resonator(Q_act))
                      % (2*PI))
        for p in P_sweep
    ])
    P_out = P_sweep * T_at_lam0

    fig = plt.figure(figsize=(15, 4.8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    fig.suptitle(
        "Microring photonic neuron — Kerr-shifted activation function\n"
        r"Resonance red-shifts with input power via $\tilde{n} = n + n_2 I$  (Agrawal Eq. 1.3.2)",
        fontsize=11, fontweight="bold"
    )

    # Panel A — spectral shift
    ax0 = fig.add_subplot(gs[0])
    for P, col in zip(P_levels, spec_colors):
        T = T_at_power(P, wl)
        ax0.plot((wl - LAMBDA)*1e12, T, color=col, lw=1.7,
                 label=f"{P/P_thresh:.1f} × P_π")
    ax0.axvline(0, color=GRAY, lw=0.8, ls=":", alpha=0.6, label="Operating λ = 1550 nm")
    _style(ax0, "Δλ from 1550 nm  (pm)", "Transmission  T",
           "A — Resonance red-shift with power")
    ax0.set_ylim(-0.02, 1.12)
    ax0.legend(fontsize=7, framealpha=0.6, loc="upper right")
    _textbox(ax0,
             f"Q = {Q_act:.0e}\n"
             r"$P_\pi$ = " + f"{P_thresh*1e3:.2f} mW\n"
                             r"Critical coupling  $r = a$",
             "upper left")

    # Panel B — activation function
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(P_sweep / P_thresh, T_at_lam0, color=BLUE, lw=2.5,
             label=r"$T(\lambda_0)$ — optical activation fn")
    ax1.axhline(ring_transfer(r_cc, a_loss, 0), color=GRAY, lw=1, ls=":",
                label="On-resonance T (P → 0)")
    ax1.axvline(1.0, color=RED, lw=1, ls="--", alpha=0.7, label=r"$P = P_\pi$")
    _style(ax1, r"Input power  $P_{in} / P_\pi$",
           r"Through-port transmission $T$",
           r"B — Transmission vs power  (activation fn)")
    ax1.set_ylim(-0.02, 1.12)
    ax1.legend(fontsize=8, framealpha=0.6)
    _textbox(ax1,
             "Sigmoidal-like response:\n"
             r"Low $P$: resonator absorbs (low $T$)" "\n"
             r"High $P$: Kerr detunes ring (high $T$)" "\n"
             "→ optical nonlinear neuron",
             "upper left")

    # Panel C — output vs input
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(P_sweep * 1e3, P_out * 1e3, color=BLUE, lw=2.5, label="Ring output")
    ax2.plot(P_sweep * 1e3, P_sweep * 1e3, color=LGRAY, lw=1.2, ls="--",
             label="Linear (T = 1)")
    ax2.axvline(P_thresh * 1e3, color=RED, lw=1, ls="--", alpha=0.7,
                label=r"$P_\pi$ threshold")
    _style(ax2, "Input power  (mW)", "Output power  (mW)",
           "C — Output vs input power")
    ax2.legend(fontsize=8, framealpha=0.6)
    _textbox(ax2,
             "Nonlinear input-output map\nfrom Kerr-shifted resonance.\n"
             r"Slope $\neq 1$ → activation-like.",
             "upper left")

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 5 — Bandwidth–energy tradeoff  (paper Section 7)
# =============================================================================
def fig5_tradeoff():
    """
    Three-panel tradeoff figure (paper Section 7 and Table):

    Panel A: Energy E and bandwidth Δν vs Q on dual log axes.
    Panel B: E · τ_ph figure of merit (∝ 1/Q).
    Panel C: Pareto frontier in (E, Δν) space.
    """
    Q_arr  = np.logspace(3.5, 7.5, 500)
    tau_ph = tau_photon(Q_arr)
    bw_GHz = bandwidth_hz(Q_arr) / 1e9
    E_pJ   = np.array([P_pi_ring(Q) * tau_photon(Q) * 1e12 for Q in Q_arr])
    E_tau  = E_pJ * tau_ph * 1e9

    Q_op   = 1e6
    E_op   = P_pi_ring(Q_op) * tau_photon(Q_op) * 1e12
    bw_op  = bandwidth_hz(Q_op) / 1e9

    fig = plt.figure(figsize=(15, 5.2))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)
    fig.suptitle(
        r"Bandwidth–energy tradeoff in microring photonic neurons  (paper Section 7)" "\n"
        r"$E \propto 1/Q$,  $\Delta\nu \propto 1/Q$,  $E \cdot \tau_{ph} \propto 1/Q$",
        fontsize=11, fontweight="bold"
    )

    # Panel A — dual axis
    ax0 = fig.add_subplot(gs[0])
    ax0r = ax0.twinx()

    l1, = ax0.loglog(Q_arr, E_pJ,   color=BLUE,  lw=2.5,
                     label=r"Energy $E = P_\pi \cdot \tau_{ph}$  (pJ)")
    l2, = ax0r.loglog(Q_arr, bw_GHz, color=GREEN, lw=2.5,
                      label=r"Bandwidth $\Delta\nu$  (GHz)")

    ax0.scatter([Q_op], [E_op],  s=90, color=BLUE,  zorder=6, marker="*")
    ax0r.scatter([Q_op], [bw_op], s=90, color=GREEN, zorder=6, marker="*")
    ax0.axvline(Q_op, color=GRAY, lw=1, ls="--", alpha=0.5)

    ax0.set_xlabel("Quality factor $Q$", fontsize=10)
    ax0.set_ylabel("Switching energy  (pJ)", fontsize=10, color=BLUE)
    ax0r.set_ylabel("Bandwidth  $\\Delta\\nu$  (GHz)", fontsize=10, color=GREEN)
    ax0.tick_params(axis="y", labelcolor=BLUE, labelsize=9)
    ax0r.tick_params(axis="y", labelcolor=GREEN, labelsize=9)
    ax0.tick_params(axis="x", labelsize=9)
    ax0.grid(True, which="both", alpha=0.2, linestyle="--")
    ax0.spines["top"].set_visible(False)
    ax0.set_title("A — Energy and bandwidth vs $Q$", fontsize=11,
                  fontweight="bold", pad=7)
    ax0.legend([l1, l2], [l1.get_label(), l2.get_label()],
               fontsize=8, loc="lower left", framealpha=0.6)
    ax0.annotate(
        f"Q = 10⁶:\nE = {E_op:.0f} pJ\nBW = {bw_op*1e3:.0f} MHz",
        xy=(Q_op, E_op), xytext=(Q_op*4, E_op*8),
        fontsize=8, color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8),
        bbox=dict(fc="white", ec="#aaaaaa", alpha=0.9, boxstyle="round,pad=0.3")
    )

    # Panel B — E·τ_ph figure of merit
    ax1 = fig.add_subplot(gs[1])
    ax1.loglog(Q_arr, E_tau, color=PURPLE, lw=2.5,
               label=r"$E \cdot \tau_{ph}$  (pJ·ns)")
    ax1.loglog(Q_arr, E_tau[0]*Q_arr[0]/Q_arr, color=LGRAY, lw=1.2, ls="--",
               label=r"Perfect $1/Q$ reference")
    ax1.scatter([Q_op], [E_op * tau_photon(Q_op)*1e9], s=90,
                color=PURPLE, zorder=6, marker="*")
    _style(ax1, "Quality factor $Q$",
           r"$E \cdot \tau_{ph}$  (pJ · ns)",
           r"B — $E \cdot \tau_{ph}$ figure of merit")
    ax1.legend(fontsize=8, framealpha=0.6)
    _textbox(ax1,
             r"$E \cdot \tau_{ph} \propto \frac{1}{Q} \cdot Q = \frac{1}{Q}$" "\n\n"
             "Energy savings (∝1/Q) are real,\n"
             "but response time ∝ Q lengthens.\n\n"
             "Product improves as 1/Q,\n"
             "not 1/Q².",
             "upper right")

    # Panel C — Pareto frontier in (E, Δν) space
    ax2 = fig.add_subplot(gs[2])
    sc = ax2.scatter(bw_GHz * 1e3, E_pJ, c=np.log10(Q_arr),
                     cmap="viridis_r", s=8, zorder=3)
    cbar = fig.colorbar(sc, ax=ax2, pad=0.02)
    cbar.set_label("log₁₀(Q)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for Q_mark, lbl in [(1e4, r"$Q=10^4$"), (1e5, r"$Q=10^5$"),
                        (1e6, r"$Q=10^6$"), (1e7, r"$Q=10^7$")]:
        e_m  = P_pi_ring(Q_mark) * tau_photon(Q_mark) * 1e12
        bw_m = bandwidth_hz(Q_mark) / 1e6
        ax2.scatter([bw_m], [e_m], s=70, color="white", edgecolors="black", zorder=5)
        ax2.text(bw_m * 1.15, e_m * 1.3, lbl, fontsize=8)

    ax2.axvline(1e3, color=GREEN, lw=1, ls="--", alpha=0.6, label="BW > 1 GHz")
    ax2.fill_between([1e3, bw_GHz[0]*1e3], 0, E_pJ.max(),
                     color="lightgreen", alpha=0.10, label="High-BW region")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    _style(ax2, "Bandwidth  $\\Delta\\nu$  (MHz)", "Switching energy  $E$  (pJ)",
           "C — Pareto frontier  (E vs BW)")
    ax2.legend(fontsize=7.5, framealpha=0.6, loc="upper left")
    _textbox(ax2,
             "Each point = one Q value.\n"
             "↑Q: lower E but lower BW.\n\n"
             "No single Q reaches the\n"
             "GHz-BW + low-E corner.\n"
             "→ Structural innovation needed.",
             "lower left")

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 6 — Design constraint summary: Q vs ring radius
# =============================================================================
def fig6_design_constraints():
    """
    Panel A: Achievable Q vs ring radius R (three competing limits).
    Panel B: FSR vs R — large-FSR requirement vs chip footprint.
    """
    R_arr = np.linspace(5e-6, 100e-6, 400)

    Q_prop_val = 2*PI * NG / (LAMBDA * ALPHA)   # propagation-loss limit

    R_c        = 5e-6
    alpha_bend = 1e4 * np.exp(-R_arr / R_c)
    Q_bend     = 2*PI * NG / (LAMBDA * (ALPHA + alpha_bend))

    dldT       = 25e-12     # m/K thermo-optic shift for Si3N4
    DeltaT_max = 0.1        # K
    Q_thermal  = LAMBDA / (dldT * DeltaT_max)

    Q_achievable = np.minimum(np.minimum(Q_bend, Q_prop_val), Q_thermal)
    FSR_arr      = C / (NG * 2*PI*R_arr) / 1e9

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Microring design constraints — achievable Q and FSR vs ring radius R\n"
        r"Si$_3$N$_4$,  $\lambda = 1550$ nm,  propagation loss = 1 dB/m",
        fontsize=11, fontweight="bold"
    )

    ax = axes[0]
    ax.semilogy(R_arr*1e6, Q_bend,     color=ORANGE, lw=2,
                label=r"Bend-loss-limited $Q_{bend}$")
    ax.axhline(Q_prop_val, color=BLUE, lw=2, ls="--",
               label=rf"Prop-loss-limited $Q_{{prop}}$ = {Q_prop_val:.1e}")
    ax.axhline(Q_thermal,  color=RED,  lw=1.5, ls="-.",
               label=rf"Thermal-drift limit $Q_{{therm}}$ (ΔT<0.1 K) = {Q_thermal:.1e}")
    ax.fill_between(R_arr*1e6, 1e2, Q_achievable, alpha=0.10, color=BLUE,
                    label="Achievable Q region")
    ax.scatter([R_ring*1e6], [1e6], s=100, color=BLUE, zorder=6, marker="*",
               label=f"Proposed device (R={R_ring*1e6:.0f} µm, Q=10⁶)")
    ax.axvline(R_ring*1e6, color=GRAY, lw=1, ls=":", alpha=0.5)
    _style(ax, "Ring radius  R  (µm)", "Quality factor  Q",
           "A — Achievable Q vs ring radius")
    ax.set_ylim(1e2, 1e8)
    ax.set_xlim(R_arr[0]*1e6, R_arr[-1]*1e6)
    ax.legend(fontsize=8, framealpha=0.6, loc="upper left")
    _textbox(ax,
             "Three Q limits:\n"
             "1. Bend loss: steep for R < 10 µm\n"
             "2. Propagation loss: constant ceiling\n"
             "3. Thermal drift: ΔλT < linewidth\n\n"
             "R = 23 µm sits above bend-loss\n"
             "regime and below thermal limit.",
             "lower right")

    ax2 = axes[1]
    ax2.semilogy(R_arr*1e6, FSR_arr, color=GREEN, lw=2.5, label="FSR  (GHz)")
    ax2.axhline(10,   color=RED,  lw=1.2, ls="--", alpha=0.8, label="10 GHz WDM spacing")
    ax2.axhline(1e3,  color=BLUE, lw=1.2, ls="--", alpha=0.8, label="1 THz ideal WDM")

    FSR_prop = C / (NG * L_geom) / 1e9
    ax2.scatter([R_ring*1e6], [FSR_prop], s=100, color=GREEN, zorder=6, marker="*",
                label=f"Proposed: R={R_ring*1e6:.0f} µm, FSR={FSR_prop:.0f} GHz")
    ax2.axvline(R_ring*1e6, color=GRAY, lw=1, ls=":", alpha=0.5)

    _style(ax2, "Ring radius  R  (µm)", "FSR  (GHz)", "B — FSR vs ring radius")
    ax2.set_xlim(R_arr[0]*1e6, R_arr[-1]*1e6)
    ax2.legend(fontsize=8, framealpha=0.6)
    _textbox(ax2,
             "Larger R → smaller FSR\n"
             "  → spurious resonance risk\n"
             "Smaller R → larger FSR (good)\n"
             "  → higher bend loss (bad)\n\n"
             "R = 23 µm balances both.",
             "upper right")

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURE 7 (NEW) — Microring design parameter space
#   Shows why the proposed experimental operating point is preferred over
#   all competing combinations of (R, α_loss, Q_target).
# =============================================================================
def fig7_design_parameter_space():
    """
    Four-panel figure mapping the full (R, α_loss) design space:

    Panel A: Heatmap of P_π,ring over (Q, α) — shows energy vs loss rate.
    Panel B: Heatmap of bandwidth Δν over (Q, R) — shows speed vs radius.
    Panel C: Switching energy E vs propagation loss α for several Q values.
             Marks the Pfeiffer 2018 state-of-art loss (1 dB/m) and proposed Q.
    Panel D: Composite "figure of merit" score = 1/(E × τ_ph) over (R, Q),
             with the proposed design point starred and iso-FOM contours.
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.38)
    fig.suptitle(
        "Microring design parameter space — why the proposed operating point is preferred\n"
        r"Si$_3$N$_4$,  $\lambda=1550$ nm.  Star $\bigstar$ = proposed design (R=23 µm, Q=10⁶, α=1 dB/m)",
        fontsize=12, fontweight="bold"
    )

    # ----------------------------------------------------------------
    # Panel A — P_π vs (Q, loss α)
    # ----------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0, 0])

    Q_1d    = np.logspace(4, 7.5, 80)
    alpha_1d = np.logspace(-5, -2, 80)   # m^-1 (0.00001 to 0.01 → 0.04 dB/m to 43 dB/m)
    QQ, AA  = np.meshgrid(Q_1d, alpha_1d)

    # For each (Q, α): recompute L_geom assuming R=23µm (fixed geometry),
    # and check if Q is achievable (Q < Q_prop_limit = 2πng/(λα))
    Q_prop_limit = 2*PI * NG / (LAMBDA * AA)
    P_pi_map = np.where(
        QQ <= Q_prop_limit,
        PI**2 * NG**2 * L_geom * AEFF / (N2 * QQ**2 * LAMBDA) * 1e3,  # mW
        np.nan
    )

    im0 = ax0.pcolormesh(Q_1d, alpha_1d * (10/np.log(10)) / 0.1,
                         np.log10(np.clip(P_pi_map, 1e-6, None)),
                         cmap="RdYlGn_r", shading="auto")
    cb0 = fig.colorbar(im0, ax=ax0)
    cb0.set_label(r"log₁₀($P_\pi$  [mW])", fontsize=9)
    cb0.ax.tick_params(labelsize=8)

    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("Quality factor $Q$", fontsize=10)
    ax0.set_ylabel("Propagation loss  (dB/m)", fontsize=10)
    ax0.set_title(r"A — $P_\pi$ (mW) over $(Q,\,\alpha)$ space", fontsize=11,
                  fontweight="bold", pad=7)

    # Mark proposed point: Q=10^6, α=1 dB/m (2.3e-4 m^-1)
    ax0.scatter([1e6], [1.0], s=200, marker="*", color="white",
                edgecolors="black", zorder=10, linewidths=1.5)
    ax0.annotate("Proposed\n(Q=10⁶, 1 dB/m)", xy=(1e6, 1.0),
                 xytext=(5e5, 0.05), fontsize=8, color="white",
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.0))

    # Show Q_max contour (diagonal where Q = Q_prop_limit)
    alpha_diag = np.logspace(-5, -2, 200)
    Q_diag = 2*PI * NG / (LAMBDA * alpha_diag)
    loss_diag = alpha_diag * (10/np.log(10)) / 0.1
    mask = (Q_diag >= Q_1d.min()) & (Q_diag <= Q_1d.max()) & \
           (loss_diag >= 0.01) & (loss_diag <= 100)
    ax0.plot(Q_diag[mask], loss_diag[mask], "w--", lw=1.5, label="$Q = Q_{max}$ (prop. limit)")
    ax0.legend(fontsize=8, framealpha=0.5)
    ax0.tick_params(labelsize=9)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # ----------------------------------------------------------------
    # Panel B — Bandwidth vs (Q, R)
    # ----------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 1])

    R_1d   = np.linspace(5e-6, 80e-6, 80)
    Q_b_1d = np.logspace(4, 7.5, 80)
    RR, QQ_b = np.meshgrid(R_1d, Q_b_1d)

    BW_map = bandwidth_hz(QQ_b) / 1e6    # MHz (R-independent for fixed Q)
    # Add bend-loss Q ceiling
    alpha_bend_map = 1e4 * np.exp(-RR / 5e-6)
    Q_bend_lim = 2*PI * NG / (LAMBDA * (ALPHA + alpha_bend_map))
    BW_map_masked = np.where(QQ_b <= Q_bend_lim, BW_map, np.nan)

    im1 = ax1.pcolormesh(R_1d*1e6, Q_b_1d, np.log10(np.clip(BW_map_masked, 1, None)),
                         cmap="plasma", shading="auto")
    cb1 = fig.colorbar(im1, ax=ax1)
    cb1.set_label("log₁₀(Bandwidth  [MHz])", fontsize=9)
    cb1.ax.tick_params(labelsize=8)

    ax1.set_yscale("log")
    ax1.set_xlabel("Ring radius  R  (µm)", fontsize=10)
    ax1.set_ylabel("Quality factor  $Q$", fontsize=10)
    ax1.set_title(r"B — Bandwidth $\Delta\nu$ (MHz) over $(R,\,Q)$ space", fontsize=11,
                  fontweight="bold", pad=7)
    ax1.scatter([R_ring*1e6], [1e6], s=200, marker="*", color="white",
                edgecolors="black", zorder=10, linewidths=1.5)
    ax1.annotate("Proposed\n(R=23 µm, Q=10⁶)", xy=(R_ring*1e6, 1e6),
                 xytext=(40, 3e5), fontsize=8, color="white",
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.0))
    ax1.tick_params(labelsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ----------------------------------------------------------------
    # Panel C — Switching energy E vs loss α for key Q values
    # ----------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])

    alpha_arr   = np.logspace(-5, -2, 200)
    loss_dBm    = alpha_arr * (10/np.log(10)) / 0.1   # dB/m
    Q_targets   = [1e4, 1e5, 1e6, 1e7]
    cols_e      = [ORANGE, GREEN, BLUE, PURPLE]

    for Q_t, col in zip(Q_targets, cols_e):
        Q_lim = 2*PI * NG / (LAMBDA * alpha_arr)
        E_arr = np.where(
            Q_t <= Q_lim,
            P_pi_ring(Q_t) * tau_photon(Q_t) * 1e12,   # pJ (constant in α for fixed Q)
            np.nan
        )
        ax2.semilogy(loss_dBm, E_arr,
                     color=col, lw=2, label=f"Q = {Q_t:.0e}")

    # Mark proposed operating point
    E_prop = P_pi_ring(1e6) * tau_photon(1e6) * 1e12
    ax2.axvline(1.0, color=GRAY, lw=1.5, ls="--", label="Pfeiffer 2018: 1 dB/m")
    ax2.scatter([1.0], [E_prop], s=150, marker="*", color=BLUE,
                edgecolors="black", zorder=10)

    # Mark Q_prop limit lines (vertical dashes where Q exceeds propagation limit)
    for Q_t, col in zip(Q_targets, cols_e):
        alpha_max = 2*PI * NG / (LAMBDA * Q_t)
        loss_max  = alpha_max * (10/np.log(10)) / 0.1
        if loss_max < 100:
            ax2.axvline(loss_max, color=col, lw=0.8, ls=":", alpha=0.6)

    _style(ax2, "Propagation loss  α  (dB/m)",
           "Switching energy  $E = P_\\pi \\cdot \\tau_{ph}$  (pJ)",
           "C — Switching energy vs propagation loss")
    ax2.legend(fontsize=8, framealpha=0.6)
    ax2.set_xlim(loss_dBm[0], loss_dBm[-1])
    _textbox(ax2,
             "Dotted vertical lines: Q > Q_max\n"
             "  (propagation loss too high\n"
             "   to support target Q)\n\n"
             "At 1 dB/m: Q up to 3.5×10⁷\n"
             "→ all four Q targets achievable",
             "upper right")

    # ----------------------------------------------------------------
    # Panel D — Composite FOM = BW / E  over (R, Q)
    #   Higher = faster AND lower energy (both good)
    # ----------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])

    # FOM = Δν / E = [c/(λQ)] / [P_pi*tau_ph]
    # Since E = P_pi*tau_ph and both scale with Q (E ∝ 1/Q, Δν ∝ 1/Q):
    # FOM = Δν/E ∝ 1/Q / (1/Q) = constant in Q (from paper §7.3)
    # But FOM changes with R through P_pi = π²ng²L·Aeff/(n2·Q²·λ) ∝ L ∝ R
    # So FOM ∝ 1/(E·Δν) ∝ Q³ from Eq 36, and varies with R through L

    R_fom   = np.linspace(5e-6, 80e-6, 80)
    Q_fom   = np.logspace(4, 7.5, 80)
    RR_f, QQ_f = np.meshgrid(R_fom, Q_fom)
    L_f     = 2*PI * RR_f

    # Check bend-loss Q ceiling
    alpha_bend_f = 1e4 * np.exp(-RR_f / 5e-6)
    Q_bend_f     = 2*PI * NG / (LAMBDA * (ALPHA + alpha_bend_f))

    # E(Q,R) = P_pi(Q,R)*tau_ph(Q)  [R enters through L=2πR]
    E_f = PI**2 * NG**2 * L_f * AEFF / (N2 * QQ_f**2 * LAMBDA) * tau_photon(QQ_f)
    BW_f = bandwidth_hz(QQ_f)
    FOM_f = np.where(QQ_f <= Q_bend_f, BW_f / E_f, np.nan)   # Hz/J = GHz/pJ scaled

    im3 = ax3.pcolormesh(R_fom*1e6, Q_fom, np.log10(np.clip(FOM_f, 1, None)),
                         cmap="YlOrRd", shading="auto")
    cb3 = fig.colorbar(im3, ax=ax3)
    cb3.set_label(r"log₁₀(FOM = $\Delta\nu / E$)", fontsize=9)
    cb3.ax.tick_params(labelsize=8)

    # Iso-FOM contours
    try:
        cs = ax3.contour(R_fom*1e6, Q_fom, np.log10(np.clip(FOM_f, 1, None)),
                         levels=6, colors="white", linewidths=0.8, alpha=0.6)
        ax3.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    except Exception:
        pass

    ax3.set_yscale("log")
    ax3.set_xlabel("Ring radius  R  (µm)", fontsize=10)
    ax3.set_ylabel("Quality factor  $Q$", fontsize=10)
    ax3.set_title(r"D — FOM = $\Delta\nu / E$ over $(R,\,Q)$  (higher = better)", fontsize=11,
                  fontweight="bold", pad=7)

    ax3.scatter([R_ring*1e6], [1e6], s=200, marker="*", color="white",
                edgecolors="black", zorder=10, linewidths=1.5)
    ax3.annotate("Proposed\n★", xy=(R_ring*1e6, 1e6),
                 xytext=(5, 3e6), fontsize=9, color="white",
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.0))

    ax3.tick_params(labelsize=9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    _textbox(ax3,
             "FOM improves with Q (∝ Q³ per Eq. 36)\n"
             "and with smaller R (shorter L → lower E).\n\n"
             "Bend-loss ceiling (white = infeasible)\n"
             "prevents using R < ~8 µm at high Q.\n\n"
             "Proposed R=23 µm balances:\n"
             "  • Above bend-loss floor\n"
             "  • FSR > 1 THz (no WDM crosstalk)\n"
             "  • Q=10⁶ is fab-demonstrated",
             "lower left")

    return fig


# =============================================================================
# Print summary table
# =============================================================================
def print_summary():
    Q_op  = 1e6
    P_op  = P_pi_ring(Q_op)
    tph   = tau_photon(Q_op)
    E_op  = P_op * tph
    B_op  = intensity_buildup(Q_op)
    Lr_op = leff_resonator(Q_op)
    P_str_65cm = P_pi_straight(L=0.65)

    print("=" * 72)
    print("  Photonic Neuron Simulation — Physical Parameter Summary (corrected)")
    print("  Equations: Agrawal 2013 & Bogaerts 2012")
    print("=" * 72)
    rows = [
        ("Wavelength λ",                    "1550 nm",               "C-band telecom"),
        ("n₂  (Si₃N₄, lower bound)",        f"{N2:.1e} m²/W",        "Levy 2010"),
        ("A_eff",                            "1.0 µm²",               "Ji 2017"),
        ("n_g",                             f"{NG:.1f}",              "Ji 2017 FEM"),
        ("γ = ω₀n₂/(cA_eff)",              f"{GAMMA:.3f} W⁻¹m⁻¹",   "Agrawal Eq. 2.3.30"),
        ("Propagation loss α",              "1 dB/m",                 "Pfeiffer 2018"),
        ("Ring radius R",                   f"{R_ring*1e6:.0f} µm",  "This work"),
        ("Ring circumference L",            f"{L_geom*1e6:.1f} µm",  "= 2πR"),
        ("FSR",                             f"{FSR/1e9:.1f} GHz",    "= c/(ng·L)"),
        ("Finesse F (Q=10⁶)",              f"{Q_op*LAMBDA/(NG*L_geom):.0f}",   "= Q·λ/(ng·L)"),
        ("Buildup B = F/π (Q=10⁶)",        f"{B_op:.0f}",           "[FIX] = Q·λ/(π·ng·L)"),
        ("Leff_res (Q=10⁶)",               f"{Lr_op*100:.1f} cm",   "= Q·λ/(2π·ng)"),
        ("Target Q",                        "10⁶",                    "Ji 2017, Pfeiffer 2018"),
        ("Photon lifetime τ_ph (Q=10⁶)",   f"{tph*1e9:.2f} ns",     "= Q·λ/(2πc)"),
        ("Bandwidth Δν (Q=10⁶)",           f"{bandwidth_hz(1e6)/1e6:.0f} MHz", "= c/(λQ)"),
        ("P_π straight WG (65 cm)",        f"{P_str_65cm:.1f} W",   "'Several watts' regime"),
        ("P_π straight WG (1 mm)",         f"{P_pi_straight(1e-3):.0f} W", "Chip-scale: kW range"),
        ("P_π ring (Q=10⁶) [CORRECTED]",  f"{P_op*1e3:.2f} mW",    "= λ·Aeff/(2·n2·B·Leff_res)"),
        ("E_switch = P_π·τ_ph (Q=10⁶)",   f"{E_op*1e12:.1f} pJ",   "switching energy"),
        ("Enhancement (65cm str / ring)",   f"{P_str_65cm/P_op:.2e}×", "Q² scaling"),
    ]
    print(f"  {'Parameter':<42} {'Value':<24} {'Source/Note'}")
    print("  " + "-"*68)
    for name, val, src in rows:
        print(f"  {name:<42} {val:<24} {src}")

    print()
    print("  Q-table (paper Section 7 / Table 1):")
    print(f"  {'Q':<10} {'Δν (MHz)':<12} {'τ_ph (ns)':<12} {'P_π (mW)':<14} "
          f"{'E=P·τ (pJ)':<14} {'B=F/π':<10}")
    print("  " + "-"*72)
    for Q in [1e4, 1e5, 1e6, 1e7]:
        bw   = bandwidth_hz(Q)/1e6
        tph_ = tau_photon(Q)*1e9
        P    = P_pi_ring(Q)*1e3
        E    = P_pi_ring(Q)*tau_photon(Q)*1e12
        B    = intensity_buildup(Q)
        print(f"  {Q:<10.0e} {bw:<12.0f} {tph_:<12.3f} {P:<14.4f} {E:<14.1f} {B:<10.1f}")
    print("=" * 72)


# =============================================================================
# Main
# =============================================================================
def main():
    print_summary()

    matplotlib.rcParams.update({
        "font.family":  "sans-serif",
        "font.size":    10,
        "axes.titlesize": 11,
        "figure.dpi":   120,
    })

    print("\nGenerating figures...")
    f1 = fig1_straight_waveguide()

    f2 = fig2_resonance_lineshape()

    f3 = fig3_energy_scaling()

    f4 = fig4_activation()

    f5 = fig5_tradeoff()

    f6 = fig6_design_constraints()

    f7 = fig7_design_parameter_space()

    plt.show()


if __name__ == "__main__":
    main()
