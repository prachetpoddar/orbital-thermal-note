"""Silicon and hot-spot terms for the die-to-radiator budget.

Companion to thermal_budget.py and spreading_stage.py (same folder).
READ THIS FIRST
  * Axisymmetric layered model, top to bottom: silicon die (radius a, adiabatic sides), the die-attach
    interface (area-specific resistance R'', local to each point), a plate of radius b, and the bonded
    joint (conductance h) on the plate's bottom face. Heat is generated on the silicon face away from the
    plate (flip-chip style: it crosses the full silicon thickness).
  * Heat flux map is two-zone: a hot core of radius c carrying a fraction f of the power, and the rest
    of the die carrying 1 - f. The core size and power fraction are PLACEHOLDERS (ASSUMED). No public
    per-region power map for any accelerator die was used.
  * Silicon thickness is ASSUMED. Silicon conductivity 130 W/m/K matches the Electronics Cooling (May 1998)
    fit k = 150 (T/300)^(-4/3) W/m/K at 60 C, reading the printed prefactor 1.5 as 150 W/m/K.
  * Still excluded: contact resistance beyond R'', package lid/substrate paths, the panel's own spreading,
    and any radiator-side effect. Isothermal panel at the fluid temperature. Totals remain lower bounds.

Solver: finite-volume, written for this file. It is validated against the series in
spreading_stage.py (by superposition of two disks) and against closed-form limits.

Run:  python hotspot_silicon.py   (self-checks first; tables print only if they pass)
"""
import math
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import spreading_stage as spr
import thermal_budget as bud

Q_W = bud.Q_W
DIE_AREA_M2 = bud.DIE_AREA_M2
FOOTPRINT_M2 = bud.JOINT_FOOTPRINT_M2
H_JOINT = bud.JOINT_H_W_M2K

K_SI = 130.0                      # W/m/K at 60 C from the Electronics Cooling fit above (about 121 at 80 C, 112 at 100 C)
T_SI_M = (0.3e-3, 0.775e-3)       # ASSUMED die thicknesses (thinned and full-thickness wafer)
TIM_R = bud.cm2k_per_w_to_m2k_per_w(bud.SOLDER_TIM_CM2K_PER_W["52In/48Sn"])   # PUBLISHED vendor value, 100 psi, 4 mil
# ASSUMED hot-spot cases: (label, core area fraction phi, core power fraction f)
HOTSPOTS = ((0.25, 0.25), (0.25, 0.50), (0.25, 0.75), (0.10, 0.30), (0.10, 0.50))   # (core area fraction, core power fraction)


def hotspot_label(phi, f):
    """Core flux relative to the die average (f/phi) and relative to the rest of the die."""
    if abs(f - phi) < 1e-12:
        return "uniform flux"
    vs_avg = f / phi
    vs_rest = f * (1 - phi) / (phi * (1 - f))
    return f"core {phi * 100:.0f}% of area, {vs_avg:.1f}x die-avg flux ({vs_rest:.1f}x the rest)"

PLATES = (("Cu 5 mm", 390.0, 5e-3), ("diamond-class 5 mm", 1000.0, 5e-3), ("k*t = 50 W/K (k_eff 10000, 5 mm)", 10000.0, 5e-3))


def _graded(length, n, ratio):
    if n < 1 or length <= 0:
        raise ValueError("bad grid segment")
    if ratio == 1.0:
        return np.full(n, length / n)
    w0 = length * (ratio - 1.0) / (ratio ** n - 1.0)
    return w0 * ratio ** np.arange(n)


def two_zone_flux(q_w, a, phi, f):
    """Core radius and the two fluxes for a core of area fraction phi carrying power fraction f."""
    if not (0 < phi < 1) or not (0 < f < 1) or q_w <= 0 or a <= 0:
        raise ValueError("bad hot-spot inputs")
    c = a * math.sqrt(phi)
    q_core = f * q_w / (math.pi * c * c)
    q_rest = (1.0 - f) * q_w / (math.pi * (a * a - c * c))
    return c, q_core, q_rest


def solve_layered(q_w, a, b, c, q_core, q_rest, t_si, k_si, r_if, t_pl, k_pl, h,
                  grid=(30, 30, 60, 12, 40), r_ratio=1.07, z_ratio=1.08):
    """Finite-volume solve. Returns dict: mean (die face), peak (die face), out (flux to fluid, W), source (W).

    Temperatures are K above the fluid. grid = (cells in [0,c], [c,a], [a,b], silicon rows, plate rows).
    """
    if not (0 < c < a < b) or min(t_si, k_si, t_pl, k_pl, h) <= 0 or r_if < 0:
        raise ValueError("bad layered inputs")
    n1, n2, n3, ns, npl = grid
    w = np.concatenate((_graded(c, n1, r_ratio)[::-1], _graded(a - c, n2, 1.0), _graded(b - a, n3, r_ratio)))
    rf = np.concatenate(([0.0], np.cumsum(w)))
    rc = 0.5 * (rf[:-1] + rf[1:])
    nr = len(w)
    ann = math.pi * (rf[1:] ** 2 - rf[:-1] ** 2)
    in_die = rf[1:] <= a * (1 + 1e-12)
    in_core = rf[1:] <= c * (1 + 1e-12)
    wz = np.concatenate((_graded(t_si, ns, 1.0), _graded(t_pl, npl, z_ratio)))
    kz = np.concatenate((np.full(ns, k_si), np.full(npl, k_pl)))
    nz = ns + npl

    exists = np.zeros((nr, nz), dtype=bool)
    for i in range(nr):
        exists[i, ns:] = True
        if in_die[i]:
            exists[i, :ns] = True
    index = -np.ones((nr, nz), dtype=int)
    count = 0
    for i in range(nr):
        for j in range(nz):
            if exists[i, j]:
                index[i, j] = count
                count += 1

    rows, cols, vals = [], [], []
    diag = np.zeros(count)
    rhs = np.zeros(count)
    source_w = 0.0

    def link(p, q_, g):
        rows.extend((p, q_))
        cols.extend((q_, p))
        vals.extend((-g, -g))
        diag[p] += g
        diag[q_] += g

    for i in range(nr):
        for j in range(nz):
            if not exists[i, j]:
                continue
            p = index[i, j]
            if i + 1 < nr and exists[i + 1, j]:
                link(p, index[i + 1, j], kz[j] * 2 * math.pi * rf[i + 1] * wz[j] / (rc[i + 1] - rc[i]))
            if j + 1 < nz and exists[i, j + 1]:
                if j == ns - 1:                      # silicon to plate across the interface resistance
                    r_tot = wz[j] / (2 * k_si) + r_if + wz[j + 1] / (2 * k_pl)
                    link(p, index[i, j + 1], ann[i] / r_tot)
                else:
                    link(p, index[i, j + 1], kz[j] * ann[i] / (0.5 * (wz[j] + wz[j + 1])))
            if j == 0:
                flux = q_core if in_core[i] else q_rest
                rhs[p] += flux * ann[i]
                source_w += flux * ann[i]
            if j == nz - 1:
                diag[p] += 1.0 / (0.5 * wz[j] / (kz[j] * ann[i]) + 1.0 / (h * ann[i]))
    rows.extend(range(count))
    cols.extend(range(count))
    vals.extend(diag)
    temp = spsolve(coo_matrix((vals, (rows, cols)), shape=(count, count)).tocsr(), rhs)

    face = np.zeros(nr)
    out = 0.0
    for i in range(nr):
        if in_die[i]:
            flux = q_core if in_core[i] else q_rest
            face[i] = temp[index[i, 0]] + flux * 0.5 * wz[0] / k_si
        out += temp[index[i, nz - 1]] / (0.5 * wz[-1] / (kz[-1] * ann[i]) + 1.0 / (h * ann[i]))
    mean = float(np.sum(face[in_die] * ann[in_die]) / np.sum(ann[in_die]))
    return {"mean": mean, "peak": float(face[in_die].max()), "out": float(out), "source": float(source_w)}


def run_case(plate_k, plate_t, t_si, phi, f, grid=(30, 30, 60, 12, 40), r_if=TIM_R, k_si=K_SI, **kw):
    a = spr.circle_radius(DIE_AREA_M2)
    b = spr.circle_radius(FOOTPRINT_M2)
    c, qc, qr = two_zone_flux(Q_W, a, phi, f)
    return solve_layered(Q_W, a, b, c, qc, qr, t_si, k_si, r_if, plate_t, plate_k, H_JOINT, grid=grid, **kw)


# ---------------------------------------------------------------- series superposition (plate-only reference)
def series_two_zone(q_w, a, b, c, f, t, k, h):
    """Plate-only series result for the two-zone flux, by superposition of two uniform disks.

    Zone fluxes: core q1 over r<c, rest q2 over c<r<a. Written as q2 over r<a plus (q1 - q2) over r<c.
    Returns (mean over die, centre).
    """
    q1 = f * q_w / (math.pi * c * c)
    q2 = (1.0 - f) * q_w / (math.pi * (a * a - c * c))
    # uniform disk of radius a at flux q2 has power q2*pi*a^2; core disk has (q1-q2)*pi*c^2
    full = spr.spreading_series(q2 * math.pi * a * a, a, b, t, k, h)
    core_power = (q1 - q2) * math.pi * c * c
    # mean over the die of the core-disk field: use the series weights for radius a (see mean_over_die)
    mean_core, centre_core = _core_disk_fields(core_power, a, b, c, t, k, h)
    return full["mean"] + mean_core, full["centre"] + centre_core


def _core_disk_fields(q_w, a, b, c, t, k, h, nterms=spr.N_SERIES):
    """Mean over r<a and centre value of the field from a uniform disk of radius c (power q_w)."""
    from scipy.special import j0, j1
    eps_c, tau, bi = c / b, t / b, h * b / k
    lam = spr.j1_zeros(nterms)
    flux = q_w / (math.pi * c * c)
    qn = 2.0 * flux * eps_c * j1(lam * eps_c) / (lam * j0(lam) ** 2)
    th = np.tanh(lam * tau)
    an_z0 = (qn * b / k) * (1.0 + (bi / lam) * th) / (lam * th + bi)
    eps_a = a / b
    mean_w = 2.0 * j1(lam * eps_a) / (lam * eps_a)
    oned = q_w / (math.pi * b * b) * (1.0 / h + t / k)
    return oned + float(np.sum(an_z0 * mean_w)), oned + float(np.sum(an_z0))


# ---------------------------------------------------------------- self-checks
def close(a, b, tol):
    return abs(a - b) <= tol * max(abs(b), 1e-30)


def run_self_checks():
    results = []

    def record(name, cond):
        results.append((name, bool(cond)))

    a = spr.circle_radius(DIE_AREA_M2)
    b = spr.circle_radius(FOOTPRINT_M2)

    # flux map accounting
    c, qc, qr = two_zone_flux(700.0, a, 0.1, 0.3)
    record("two-zone flux conserves the power", close(qc * math.pi * c * c + qr * math.pi * (a * a - c * c), 700.0, 1e-12))
    record("uniform case (phi = f) gives equal fluxes", close(two_zone_flux(700.0, a, 0.25, 0.25)[1], two_zone_flux(700.0, a, 0.25, 0.25)[2], 1e-12))

    # (1) layered solver vs the plate-only series: vanishing silicon, no interface, uniform flux
    thin = dict(t_si=1e-6, r_if=0.0)
    for name, k, t in (("Cu 5 mm", 390.0, 5e-3), ("Al 2 mm", 167.0, 2e-3)):
        fv = run_case(k, t, phi=0.25, f=0.25, **thin)
        ser = spr.spreading_series(700.0, a, b, t, k, H_JOINT)
        record(f"layered solver matches the plate series, uniform flux, thin silicon: {name} (mean, 0.7%)", close(fv["mean"], ser["mean"], 7e-3))
        record(f"layered solver matches the plate series, uniform flux, thin silicon: {name} (peak, 1.5%)", close(fv["peak"], ser["centre"], 1.5e-2))
    # (2) the same against superposition for a hot spot (two different hot-spot shapes)
    for phi, f in ((0.25, 0.75), (0.10, 0.50)):
        fv = run_case(390.0, 5e-3, phi=phi, f=f, **thin)
        c_, _, _ = two_zone_flux(700.0, a, phi, f)
        mean_s, centre_s = series_two_zone(700.0, a, b, c_, f, 5e-3, 390.0, H_JOINT)
        record(f"layered solver matches series superposition, hot spot phi={phi}, f={f} (mean, 0.7%)", close(fv["mean"], mean_s, 7e-3))
        record(f"layered solver matches series superposition, hot spot phi={phi}, f={f} (peak, 1.5%)", close(fv["peak"], centre_s, 1.5e-2))

    # (3) silicon and interface in a limit with a closed form: plate and silicon both nearly isothermal laterally
    q_avg = 700.0 / DIE_AREA_M2
    huge = 1e7
    fv = run_case(huge, 5e-3, t_si=0.775e-3, phi=0.25, f=0.25, r_if=3.9e-6, k_si=huge)
    expect = q_avg * 3.9e-6 + 700.0 / (math.pi * b * b * H_JOINT)
    record("isothermal silicon and plate: die rise equals q*R'' + Q/(pi b^2 h)", close(fv["mean"], expect, 5e-3))
    fv2 = run_case(huge, 5e-3, t_si=0.775e-3, phi=0.25, f=0.25, r_if=3.9e-6 + 2e-6, k_si=huge)
    record("isothermal limit: adding 2e-6 m2K/W of interface adds q*2e-6", close(fv2["mean"] - fv["mean"], q_avg * 2e-6, 2e-2))
    # silicon alone, plate effectively isothermal: uniform flux through t/k plus the interface
    fv3 = run_case(huge, 5e-3, t_si=0.775e-3, phi=0.25, f=0.25, r_if=3.9e-6, k_si=K_SI)
    expect3 = q_avg * (0.775e-3 / K_SI + 3.9e-6) + 700.0 / (math.pi * b * b * H_JOINT)
    record("uniform flux through real silicon with an isothermal plate: q*(t/k + R'') + joint", close(fv3["mean"], expect3, 5e-3))

    # (4) energy and convergence
    base = run_case(390.0, 5e-3, t_si=0.775e-3, phi=0.10, f=0.50)
    record("source power equals 700 W", close(base["source"], 700.0, 1e-9))
    record("energy leaving the bottom equals the source (1e-6)", close(base["out"], base["source"], 1e-6))
    fine = run_case(390.0, 5e-3, t_si=0.775e-3, phi=0.10, f=0.50, grid=(60, 60, 120, 24, 80), r_ratio=1.035, z_ratio=1.04)
    record("mean changes under 0.5% on a 2x refined grid", close(base["mean"], fine["mean"], 5e-3))
    record("peak changes under 1% on a 2x refined grid", close(base["peak"], fine["peak"], 1e-2))

    # (5) physical ordering
    uni = run_case(390.0, 5e-3, t_si=0.775e-3, phi=0.25, f=0.25)
    hot = run_case(390.0, 5e-3, t_si=0.775e-3, phi=0.25, f=0.75)
    record("a hot spot raises the peak", hot["peak"] > uni["peak"])
    record("a hot spot raises the mean", hot["mean"] > uni["mean"])
    record("peak is above mean", hot["peak"] > hot["mean"])
    thin_si = run_case(390.0, 5e-3, t_si=0.3e-3, phi=0.25, f=0.75)
    record("thinner silicon lowers the hot-spot peak", thin_si["peak"] < hot["peak"])
    record("silicon adds to the uniform-flux rise", uni["mean"] > run_case(390.0, 5e-3, t_si=1e-6, phi=0.25, f=0.25)["mean"])
    record("a higher plate conductivity lowers the hot-spot peak",
           run_case(1000.0, 5e-3, t_si=0.775e-3, phi=0.25, f=0.75)["peak"] < hot["peak"])

    # (6) the check against the earlier script's whole-stack total (uniform flux, thin silicon)
    d = spr.stack_drops(700.0, DIE_AREA_M2, FOOTPRINT_M2, 5e-3, 390.0, 0.039)
    fv_u = run_case(390.0, 5e-3, t_si=1e-6, phi=0.25, f=0.25)
    record("thin-silicon uniform case reproduces the earlier stack total (within 1%)", close(fv_u["mean"], d["total"], 1e-2))

    # (6b) regression anchors from an independent solver (written by a separate reviewer from the written spec,
    #      converged to ~0.01%; the reviewer used a = 0.016095, b = 0.11284, h = 4030, which differ from this
    #      script's constants by about 0.05%). Tolerance 0.2% so lateral-conduction errors above about 5% are caught.
    anchors = (
        ("case 1: Cu 5 mm, Si 0.775 mm, core 10%/50%", (390.0, 5e-3, 0.775e-3, 0.10, 0.50), 65.898, 140.561),
        ("case 2: k*t 50 plate, Si 0.775 mm, core 25%/75%", (10000.0, 5e-3, 0.775e-3, 0.25, 0.75), 16.346, 34.8155),
        ("case 3: Cu 5 mm, Si 0.3 mm, uniform", (390.0, 5e-3, 0.3e-3, 0.25, 0.25), 58.889, 69.595),
        ("case 4: k 1000 plate, Si 0.775 mm, core 10%/30%", (1000.0, 5e-3, 0.775e-3, 0.10, 0.30), 38.7585, 66.668),
    )
    for name, args, mean_ref, peak_ref in anchors:
        rr = run_case(*args)
        record(f"independent-solver anchor, mean (0.2%): {name}", close(rr["mean"], mean_ref, 2e-3))
        record(f"independent-solver anchor, peak (0.2%): {name}", close(rr["peak"], peak_ref, 2e-3))

    # (7) invalid inputs raise
    for label, call in [
        ("core fraction above one raises", lambda: two_zone_flux(700.0, 0.016, 1.5, 0.5)),
        ("power fraction of zero raises", lambda: two_zone_flux(700.0, 0.016, 0.1, 0.0)),
        ("core larger than the die raises", lambda: solve_layered(700.0, 0.016, 0.11, 0.02, 1.0, 1.0, 1e-3, 130.0, 0.0, 5e-3, 390.0, 4030.0)),
        ("die larger than the footprint raises", lambda: solve_layered(700.0, 0.2, 0.11, 0.01, 1.0, 1.0, 1e-3, 130.0, 0.0, 5e-3, 390.0, 4030.0)),
        ("negative interface resistance raises", lambda: run_case(390.0, 5e-3, t_si=1e-3, phi=0.1, f=0.3, r_if=-1.0)),
    ]:
        try:
            call()
            record(label, False)
        except ValueError:
            record(label, True)
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

    print("INPUTS")
    print(f"  Q = {Q_W:.0f} W; die {DIE_AREA_M2 * 1e6:.0f} mm2; footprint {FOOTPRINT_M2 * 1e4:.0f} cm2; joint h = {H_JOINT:.0f} W/m2/K")
    print(f"  silicon k = {K_SI} W/m/K (Electronics Cooling fit at 60 C), thickness {', '.join(f'{t * 1e3:.3f}' for t in T_SI_M)} mm (ASSUMED)")
    print(f"  die-attach interface R'' = {TIM_R:.2e} m2K/W (52In/48Sn, Indium brochure, vendor value), applied locally")
    print("  hot-spot sizes and power shares are PLACEHOLDERS; no public power map was used")
    print("  SCOPE: axisymmetric, isothermal panel, no contact resistance beyond R''. Still a lower bound.")

    print("\nTABLE H1  uniform flux: what the silicon adds. Mean die rise above the panel, K.")
    print("  plate                                | silicon mm | no silicon (earlier) | with silicon | silicon adds")
    for pname, k, t in PLATES:
        d = spr.stack_drops(Q_W, DIE_AREA_M2, FOOTPRINT_M2, t, k, 0.039)
        for ts in T_SI_M:
            r = run_case(k, t, ts, 0.25, 0.25)
            print(f"  {pname:36s} | {ts * 1e3:10.3f} | {d['total']:20.2f} | {r['mean']:12.2f} | {r['mean'] - d['total']:12.2f}")

    print("\nTABLE H2  hot spots, silicon 0.775 mm. Die-face rise above the panel, K. 'peak' = die-face temperature at the centre of the hot core.")
    print("  'area x mean' = radiator area vs zero drop when the MEAN die face is held at 60 C (q_abs = 0). 'peak at' = peak die-face temperature in C")
    print("  if the panel is sized for that mean-at-60-C design (i.e. 60 + peak - mean). No peak area multiple is shown: it would compare different designs.")
    print("  plate / case                                                     | mean  | peak  | peak - mean | area x mean | peak at (C)")
    for pname, k, t in PLATES:
        for phi, f in HOTSPOTS:
            r = run_case(k, t, 0.775e-3, phi, f)
            print(f"  {pname.split(' (')[0][:20]:20s} {hotspot_label(phi, f):50s} | {r['mean']:5.1f} | {r['peak']:5.1f} | {r['peak'] - r['mean']:11.1f} | "
                  f"{spr.mult_str(spr.area_multiple(r['mean'])):>11s} | {60.0 + r['peak'] - r['mean']:11.1f}")

    print("\nTABLE H3  silicon thickness, core 10% of area carrying 50% of the power (5x die-avg flux). Peak rise above the panel, K.")
    print("  plate                              | 0.100 mm | 0.300 mm | 0.775 mm")
    for pname, k, t in PLATES:
        vals = [run_case(k, t, ts, 0.10, 0.50)["peak"] for ts in (0.1e-3, 0.3e-3, 0.775e-3)]
        print(f"  {pname:34s} | {vals[0]:8.1f} | {vals[1]:8.1f} | {vals[2]:8.1f}")

    print("\nTABLE H4  silicon conductivity sensitivity (hot silicon conducts less). Core 10% / 50% power, silicon 0.775 mm. Peak rise, K.")
    print("  plate                              | k_si 130 | k_si 100 | k_si 80")
    for pname, k, t in PLATES:
        vals = [run_case(k, t, 0.775e-3, 0.10, 0.50, k_si=ks)["peak"] for ks in (130.0, 100.0, 80.0)]
        print(f"  {pname:34s} | {vals[0]:8.1f} | {vals[1]:8.1f} | {vals[2]:8.1f}")
    print("  k_si falls with temperature (as T^(-4/3) in the Electronics Cooling fit: about 121 W/m/K at 80 C, 112 at 100 C); the model uses one value.")

    print("\nSELF-CHECKS")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  {len(checks)}/{len(checks)} passed")
    print("  Verification stops here: the solver is checked against the earlier series (by superposition), closed-form limits, and four")
    print("  numbers from an independent solver. Not covered: several hot spots, neighbouring dies, temperature-dependent conductivity.")
    print("  Table rows are not individually tested. Hot-spot sizes are placeholders and say nothing about any real chip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
