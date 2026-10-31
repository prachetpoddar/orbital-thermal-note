"""Sensitivity of radiator area to the conduction temperature drop between an
accelerator die and a radiator, for a 700 W-class chip in orbit.

READ THIS FIRST
  * This is a sensitivity study, not a design tool. Every stack total printed
    here EXCLUDES the heat-spreading stage between the 8.14 cm2 die and the
    joint footprint, which is likely the largest term. Totals are lower bounds.
  * "T_j" here is the die-surface temperature at the start of the chain. No
    silicon or junction-to-case resistance is modelled.
  * The sink is ideal deep space (q_abs = 0) unless stated. That is optimistic.
  * Every input is tagged PUBLISHED (with source) or ASSUMED. A PUBLISHED tag
    means a number appears in the cited source, not that it is a measurement of
    the stack being modelled.

Model:
    T_rad  = T_j - dT_stack
    A_rad  = Q / (sides * eta * (eps * SIGMA * T_rad**4 - q_abs))
Radiated power scales as T**4, so dA/A = -4 dT / T_rad (first order).
Two ways to read a drop dT:
    (a) T_j rise at a FIXED radiator temperature: exactly dT.  Use this for big drops.
    (b) radiator area penalty at a FIXED T_j: meaningful only for small drops
        (printed as n/m above N_M_THRESHOLD_K).

Run:  python thermal_budget.py   (self-checks run first; tables print only if they pass)
"""
import sys

SIGMA = 5.670374419e-8  # W m-2 K-4, CODATA 2018
N_M_THRESHOLD_K = 15.0  # above this drop the fixed-T_j area penalty is shown as n/m

# ---------------------------------------------------------------- inputs
Q_W = 700.0              # PUBLISHED: H100 SXM max TDP "Up to 700W (configurable)", NVIDIA H100 datasheet (Sep 2022). Read 2026-10-31.
DIE_AREA_M2 = 814e-6     # PUBLISHED: GH100 die area 814 mm2, TechPowerUp GPU database (H100 SXM5 80 GB). Not an NVIDIA figure. Read 2026-10-31.
EPS = 0.85               # ASSUMED, inside the usual range for white coatings (about 0.8 to 0.9)
ETA = 0.95               # PUBLISHED claim, ACT: CCHP-embedded radiator panels "over 95%" efficiency (vendor claim)
Q_ABS = 0.0              # ASSUMED: ideal deep-space sink, no absorbed solar or Earth flux (optimistic)
Q_ABS_SWEEP = (0.0, 100.0, 200.0)  # ASSUMED: absorbed flux sweep, W/m2, applied to the radiating face
T_J_BASE = 333.15        # 60 C reference: IEEE Spectrum (Cavalier, 11 June 2026) models a 700 W chip held at a constant 60 C. Read 2026-10-31.
T_J_HOT = 353.15         # ASSUMED 80 C, second scenario
PANEL_GRADIENT_K = 3.0   # ASSUMED, from ACT's qualitative "a few degrees C". OVERLAPS with ETA: fin efficiency
                         # already encodes in-panel non-uniformity, so adding both double-counts. Shown separately.

# Indium Corp TIMs brochure 99342 (thermal resistance at 100 psi, 0.004 in bond line):
SOLDER_TIM_CM2K_PER_W = {"52In/48Sn": 0.0390, "99.99In": 0.0514}   # PUBLISHED, vendor datasheet values
JOINT_W_PER_IN2_K = 2.6                          # PUBLISHED: "design value" for the bonded embedded-heat-pipe joint
                                                 # (patent US 12515822). Reference area (pipe contact vs panel) NOT confirmed.
JOINT_H_W_M2K = JOINT_W_PER_IN2_K / (0.0254 ** 2)
JOINT_FOOTPRINT_M2 = 400e-4                      # ASSUMED 400 cm2 joint footprint used in the worked stacks
POLYMER_K = (0.5, 1.0, 5.0)                      # 5 = PUBLISHED upper bound for non-metallic TIMs, 1 = published "typical
                                                 # is below about 1" (patent US 7663242), 0.5 = ASSUMED below-typical case
POLYMER_BLT_M = (25e-6, 50e-6, 100e-6)           # ASSUMED bond-line thicknesses; bulk BLT/k only, no contact resistance


def cm2k_per_w_to_m2k_per_w(r):
    return r * 1e-4


# ---------------------------------------------------------------- model
def radiator_area(q_w, t_rad_k, eps=EPS, eta=ETA, sides=1, q_abs=Q_ABS):
    """Radiator area in m2 needed to reject q_w watts at mean radiator temperature t_rad_k."""
    if q_w <= 0:
        raise ValueError("q_w must be positive")
    if t_rad_k <= 0:
        raise ValueError("t_rad_k must be positive kelvin")
    if not (0 < eps <= 1) or not (0 < eta <= 1):
        raise ValueError("eps and eta must be in (0, 1]")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    net = eps * SIGMA * t_rad_k ** 4 - q_abs
    if net <= 0:
        raise ValueError("net emitted flux is not positive")
    return q_w / (sides * eta * net)


def drop_area_specific(q_w, area_m2, r_spec_m2k_per_w):
    """Temperature drop (K) across a layer with area-specific resistance r_spec over area_m2."""
    if q_w <= 0 or area_m2 <= 0 or r_spec_m2k_per_w < 0:
        raise ValueError("bad layer inputs")
    return (q_w / area_m2) * r_spec_m2k_per_w


def drop_polymer_tim(q_w, area_m2, k_w_mk, blt_m):
    if k_w_mk <= 0 or blt_m <= 0:
        raise ValueError("bad TIM inputs")
    return drop_area_specific(q_w, area_m2, blt_m / k_w_mk)


def drop_joint(q_w, footprint_m2, h_w_m2k=None):
    h = JOINT_H_W_M2K if h_w_m2k is None else h_w_m2k
    if h <= 0:
        raise ValueError("joint conductance must be positive")
    return drop_area_specific(q_w, footprint_m2, 1.0 / h)


def stack_total(*drops):
    if any(d < 0 for d in drops):
        raise ValueError("negative drop")
    return sum(drops)


def area_penalty_pct(t_j_k, d_t_k, **kw):
    """Percent more radiator area than a zero-drop stack at the same T_j."""
    base = radiator_area(Q_W, t_j_k, **kw)
    return 100.0 * (radiator_area(Q_W, t_j_k - d_t_k, **kw) / base - 1.0)


def penalty_str(d_t_k, t_j_k=T_J_BASE, **kw):
    if d_t_k > N_M_THRESHOLD_K:
        return "n/m"
    return f"{area_penalty_pct(t_j_k, d_t_k, **kw):.1f}"


def footprint_for_joint_drop(q_w, d_t_k):
    """Footprint (m2) at which the bonded joint alone drops d_t_k kelvin."""
    return q_w / (JOINT_H_W_M2K * d_t_k)


# ---------------------------------------------------------------- self-checks
def sensitivity_ok(fn, t=330.0, h=0.01, tol=2e-3):
    """True if fn(t) obeys the T**4 law: relative slope equals -4/t within tol (relative)."""
    rel = (fn(t + h) - fn(t - h)) / (2 * h) / fn(t)
    return abs(rel / (-4.0 / t) - 1.0) < tol


def energy_balance_ok(fn, q_w=700.0, t=330.0, eps=EPS, eta=ETA, sides=1, q_abs=0.0, tol=1e-9):
    """True if net radiated power at the returned area equals q_w."""
    a = fn(q_w, t, eps=eps, eta=eta, sides=sides, q_abs=q_abs)
    return abs(sides * eta * (eps * SIGMA * t ** 4 - q_abs) * a - q_w) / q_w < tol


def joint_conversion_ok(h_w_m2k):
    """True if h converts 2.6 W per square inch per K, using 1 in2 = 6.4516e-4 m2 exactly."""
    return abs(h_w_m2k * 6.4516e-4 - JOINT_W_PER_IN2_K) < 1e-9


def _buggy_t3(q_w, t, eps=EPS, eta=ETA, sides=1, q_abs=0.0):
    return q_w / (sides * eta * eps * SIGMA * t ** 3)


def _buggy_no_eta(q_w, t, eps=EPS, eta=ETA, sides=1, q_abs=0.0):
    return q_w / (sides * eps * SIGMA * t ** 4)


def _buggy_ignores_qabs(q_w, t, eps=EPS, eta=ETA, sides=1, q_abs=0.0):
    return q_w / (sides * eta * eps * SIGMA * t ** 4)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def run_self_checks():
    results = []

    def record(name, cond):
        results.append((name, bool(cond)))

    # radiation law: checks must pass on the model and FAIL on planted bad models
    record("T^4 sensitivity passes on the model", sensitivity_ok(lambda t: radiator_area(Q_W, t)))
    record("T^4 sensitivity fires on a T^3 model", not sensitivity_ok(lambda t: _buggy_t3(Q_W, t)))
    record("energy balance passes on the model", energy_balance_ok(radiator_area))
    record("energy balance fires on a model that drops eta", not energy_balance_ok(_buggy_no_eta))
    record("energy balance passes with two sides", energy_balance_ok(radiator_area, sides=2))
    record("energy balance passes with absorbed flux", energy_balance_ok(radiator_area, q_abs=150.0))
    record("energy balance fires on a model that ignores absorbed flux",
           not energy_balance_ok(_buggy_ignores_qabs, q_abs=150.0))
    record("absorbed flux increases the area",
           radiator_area(Q_W, 330.0, q_abs=100.0) > radiator_area(Q_W, 330.0))

    # unit conversions, applied to the model constants themselves
    record("model joint conductance converts 2.6 W/in2/K correctly", joint_conversion_ok(JOINT_H_W_M2K))
    record("conversion check fires on 0.0254 not squared", not joint_conversion_ok(JOINT_W_PER_IN2_K / 0.0254))
    record("conversion check fires on a 39.37 factor instead of 39.37^2", not joint_conversion_ok(2.6 * 39.37))
    record("cm2 K/W to m2 K/W: 0.039 becomes 3.9e-6",
           close(cm2k_per_w_to_m2k_per_w(SOLDER_TIM_CM2K_PER_W["52In/48Sn"]), 3.9e-6))

    # known-answer tests for the layer functions (hand-computed, independent of the model constants)
    record("area-specific drop: 100 W over 0.01 m2 at 1e-3 m2K/W is 10 K", close(drop_area_specific(100.0, 0.01, 1e-3), 10.0))
    record("polymer drop: 100 W, 0.01 m2, k=1, 1 mm is 10 K", close(drop_polymer_tim(100.0, 0.01, 1.0, 1e-3), 10.0))
    record("polymer drop doubles when thickness doubles",
           close(drop_polymer_tim(700.0, 814e-6, 1.0, 100e-6), 2 * drop_polymer_tim(700.0, 814e-6, 1.0, 50e-6)))
    record("joint drop: a power of h*A*5 gives 5 K", close(drop_joint(1000.0 * 0.01 * 5.0, 0.01, h_w_m2k=1000.0), 5.0))
    record("joint drop falls as 1/footprint",
           close(drop_joint(700.0, 100e-4), 4 * drop_joint(700.0, 400e-4)))
    record("footprint for a joint drop round-trips",
           close(drop_joint(700.0, footprint_for_joint_drop(700.0, 5.0)), 5.0))
    record("stack total is the plain sum", close(stack_total(1.5, 2.0, 3.0), 6.5))
    record("fixed-T_j penalty of zero drop is zero", abs(area_penalty_pct(T_J_BASE, 0.0)) < 1e-12)
    record("fixed-T_j penalty matches a hand value: 1 K at 333.15 K is (333.15/332.15)^4 - 1",
           close(area_penalty_pct(T_J_BASE, 1.0) / 100.0, (333.15 / 332.15) ** 4 - 1.0, tol=1e-9))

    # invalid inputs must raise
    for label, call in [
        ("negative power raises", lambda: radiator_area(-1.0, 330.0)),
        ("zero temperature raises", lambda: radiator_area(700.0, 0.0)),
        ("emissivity above 1 raises", lambda: radiator_area(700.0, 330.0, eps=1.2)),
        ("three sides raises", lambda: radiator_area(700.0, 330.0, sides=3)),
        ("absorbed flux above emission raises", lambda: radiator_area(700.0, 200.0, q_abs=1e4)),
        ("zero footprint raises", lambda: drop_joint(700.0, 0.0)),
        ("negative drop in a stack raises", lambda: stack_total(1.0, -2.0)),
    ]:
        try:
            call()
            record(label, False)
        except ValueError:
            record(label, True)

    # ordering behaviour
    record("larger drop means larger area", radiator_area(Q_W, 320.0) > radiator_area(Q_W, 330.0))
    record("two sides halve the area",
           close(radiator_area(Q_W, 330.0, sides=2) * 2, radiator_area(Q_W, 330.0), tol=1e-12))
    first_order = 100 * 4 * 10 / T_J_BASE
    exact = area_penalty_pct(T_J_BASE, 10.0)
    record("exact 10 K penalty is above the first-order estimate and within 20 percent of it",
           first_order < exact < first_order * 1.2)
    return results


# ---------------------------------------------------------------- tables
def main():
    checks = run_self_checks()
    failed = [n for n, ok in checks if not ok]
    if failed:
        print("SELF-CHECKS FAILED, tables not printed:")
        for n in failed:
            print(f"  [FAIL] {n}")
        return 1

    die_flux = Q_W / DIE_AREA_M2
    print("INPUTS (tags in the source file)")
    print(f"  Q = {Q_W:.0f} W, die = {DIE_AREA_M2 * 1e6:.0f} mm2, die flux = {die_flux:.4g} W/m2")
    print(f"  eps = {EPS}, eta = {ETA}, absorbed flux = {Q_ABS} W/m2 (sweep in Table 6), sink 0 K")
    print(f"  joint conductance = {JOINT_H_W_M2K:.1f} W/m2/K (2.6 W/in2/K); assumed joint footprint = {JOINT_FOOTPRINT_M2 * 1e4:.0f} cm2")
    for alloy, r in SOLDER_TIM_CM2K_PER_W.items():
        print(f"  solder TIM {alloy}: {r} cm2 C/W = {cm2k_per_w_to_m2k_per_w(r):.2e} m2 K/W")
    print(f"  T_j reference: {T_J_BASE - 273.15:.0f} C and {T_J_HOT - 273.15:.0f} C (die-surface, no silicon resistance)")
    print("  SCOPE: no spreading stage between die and joint footprint; all totals are lower bounds")

    print("\nTABLE 1  baseline radiator area per chip, zero conduction drop, q_abs = 0")
    for label, tj in (("60 C", T_J_BASE), ("80 C", T_J_HOT)):
        print(f"  T_rad = {label}: one side {radiator_area(Q_W, tj):.3f} m2, two sides {radiator_area(Q_W, tj, sides=2):.3f} m2")

    print("\nTABLE 2  radiator area penalty at fixed T_j vs conduction drop (q_abs = 0, same % for one or two sides)")
    print("  dT_K | T_rad(60C) | penalty%(60C) | penalty%(80C) | first-order%(60C)")
    for d in (0, 1, 2, 3, 5, 10, 15, 20):
        print(f"  {d:4d} | {T_J_BASE - d:9.2f}  | {area_penalty_pct(T_J_BASE, d):12.2f}  | "
              f"{area_penalty_pct(T_J_HOT, d):12.2f}  | {100 * 4 * d / T_J_BASE:14.2f}")

    print(f"\nTABLE 3  die-to-lid TIM at die-scale flux (die area {DIE_AREA_M2 * 1e6:.0f} mm2). "
          f"Column 2 is the T_j rise at fixed radiator temperature; column 3 is the fixed-T_j area penalty (n/m above {N_M_THRESHOLD_K:.0f} K)")
    print("  option                              | T_j rise K | area penalty %")
    for alloy, r in SOLDER_TIM_CM2K_PER_W.items():
        d = drop_area_specific(Q_W, DIE_AREA_M2, cm2k_per_w_to_m2k_per_w(r))
        print(f"  solder TIM {alloy:10s} {r:.4f} cm2C/W | {d:10.2f} | {penalty_str(d):>8s}")
    for k in POLYMER_K:
        for blt in POLYMER_BLT_M:
            d = drop_polymer_tim(Q_W, DIE_AREA_M2, k, blt)
            print(f"  polymer k={k:3.1f} W/mK, BLT={blt * 1e6:3.0f} um        | {d:10.2f} | {penalty_str(d):>8s}")

    print("\nTABLE 4  bonded joint to heat-pipe panel vs footprint (2.6 W/in2/K; reference area unconfirmed)")
    print("  footprint_cm2 | T_j rise K | area penalty %")
    for fp_cm2 in (12, 100, 400, 900):
        d = drop_joint(Q_W, fp_cm2 * 1e-4)
        print(f"  {fp_cm2:12d}  | {d:10.2f} | {penalty_str(d):>8s}")
    for target in (2.0, 5.0):
        print(f"  footprint for a {target:.0f} K joint drop: {footprint_for_joint_drop(Q_W, target) * 1e4:.0f} cm2")

    print("\nTABLE 5  worked stacks, spreading stage NOT included (lower bounds). Joint footprint 400 cm2 (ASSUMED).")
    joint = drop_joint(Q_W, JOINT_FOOTPRINT_M2)
    rows = []
    for alloy, r in SOLDER_TIM_CM2K_PER_W.items():
        rows.append((f"solder {alloy}", drop_area_specific(Q_W, DIE_AREA_M2, cm2k_per_w_to_m2k_per_w(r))))
    rows.append(("polymer k=5 (patent upper bound) x 50 um", drop_polymer_tim(Q_W, DIE_AREA_M2, 5.0, 50e-6)))
    rows.append(("polymer k=1 (patent typical) x 50 um", drop_polymer_tim(Q_W, DIE_AREA_M2, 1.0, 50e-6)))
    print("  stack                                   | TIM    joint  | interfaces K | penalty% | + 3 K panel gradient K | penalty%")
    for label, d_tim in rows:
        t_if = stack_total(d_tim, joint)
        t_pg = stack_total(d_tim, joint, PANEL_GRADIENT_K)
        print(f"  {label:40s}| {d_tim:5.2f}  {joint:5.2f}  | {t_if:11.2f}  | {penalty_str(t_if):>8s} | {t_pg:21.2f}  | {penalty_str(t_pg):>8s}")
    print("  (the panel gradient overlaps with eta = 0.95, so the right-hand pair is an upper reading)")

    print("\nTABLE 6  q_abs sweep: baseline area and fixed-T_j penalty (60 C), one side. Absorbed flux on the radiating face.")
    t_if_ref = stack_total(drop_area_specific(Q_W, DIE_AREA_M2, cm2k_per_w_to_m2k_per_w(SOLDER_TIM_CM2K_PER_W["52In/48Sn"])), joint)
    print(f"  stack used: solder 52In/48Sn + joint 400 cm2, interfaces only = {t_if_ref:.2f} K")
    print("  q_abs W/m2 | baseline area m2 | penalty% for that stack | penalty% for 5 K")
    for qa in Q_ABS_SWEEP:
        print(f"  {qa:9.0f}  | {radiator_area(Q_W, T_J_BASE, q_abs=qa):15.3f}  | "
              f"{area_penalty_pct(T_J_BASE, t_if_ref, q_abs=qa):22.1f}  | {area_penalty_pct(T_J_BASE, 5.0, q_abs=qa):15.1f}")

    print("\nSELF-CHECKS")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  {len(checks)}/{len(checks)} passed")
    print("  Verification stops here: the checks cover internal consistency, unit handling and hand-computed layer values, "
          "not the physics of any real stack.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
