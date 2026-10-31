"""Spreading stage between a die and a bonded radiator joint: the term the first
budget script left out.

Companion to thermal_budget.py (same folder). READ THIS FIRST
  * Geometry is an axisymmetric idealisation: a uniform-flux disk source (the die,
    equal-area circle) on the top face of a circular plate (the lid/spreader,
    equal-area circle for the joint footprint). Adiabatic sides and adiabatic
    top outside the die. The bottom face sees the bonded-joint conductance h.
  * Real dies are square, have hot spots, and share a footprint with neighbours.
    Uniform flux makes this OPTIMISTIC. The adiabatic outer edge is equivalent to a periodic array of
    identical chips at this footprint pitch. Mean die temperature is used;
    the peak (die centre) is also printed.
  * Plate conductivities are handbook values and plate thickness / footprint are
    chosen, not measured. All tagged ASSUMED below.
  * The joint conductance (2.6 W/in2/K) is a patent design value whose reference
    area is unconfirmed. See thermal_budget.py.
  * Excludes: silicon / junction-to-case resistance, spreader-to-lid TIM on the
    plate's top face, contact resistance anywhere, and the radiator panel's own gradient.

Method:
  Series solution (Bessel J0 eigenfunctions, J1(lambda)=0) for the plate, with a
  separate axisymmetric finite-volume solver as an independent cross-check.

Run:  python spreading_stage.py   (self-checks first; tables print only if they pass)
"""
import math
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import j0, j1, jn_zeros

import thermal_budget as bud

Q_W = bud.Q_W
DIE_AREA_M2 = bud.DIE_AREA_M2
H_JOINT = bud.JOINT_H_W_M2K
FOOTPRINT_M2 = bud.JOINT_FOOTPRINT_M2            # ASSUMED 400 cm2

# Handbook conductivities, W/m/K, read 2026-10-31: Al 6061-T6 167 (nTop material sheet citing MatWeb);
# Cu C10100 226 Btu/(ft h F) at 68 F = 391 (Copper Development Association), rounded to 390;
# diamond-class 1000 is an ASSUMED round figure for a high-conductivity composite, not a sourced value.
K_MATERIALS = (("Al 6061", 167.0), ("Cu", 390.0), ("diamond-class", 1000.0))
T_PLATE_M = (2e-3, 5e-3, 10e-3)                  # ASSUMED plate thicknesses
FOOTPRINTS_CM2 = (100, 200, 400, 900)           # ASSUMED sweep

N_SERIES = 4000


def circle_radius(area_m2):
    if area_m2 <= 0:
        raise ValueError("area must be positive")
    return math.sqrt(area_m2 / math.pi)


# ---------------------------------------------------------------- series solution
_ZEROS = {}


def j1_zeros(n):
    if n not in _ZEROS:
        _ZEROS[n] = jn_zeros(1, n)
    return _ZEROS[n]


def spreading_series(q_w, a, b, t, k, h, nterms=N_SERIES):
    """Mean and centre temperature rise (K above the far-side fluid) of the source disk.

    Returns dict with keys: mean, centre, oned_plate, oned_joint, spreading_mean.
    Plate: radius b, thickness t, conductivity k. Source: disk radius a <= b on top face,
    uniform flux. Bottom face: convective conductance h to a fluid at 0 K.
    """
    if not (0 < a <= b) or t <= 0 or k <= 0 or h <= 0 or q_w <= 0:
        raise ValueError("bad spreading inputs")
    eps = a / b
    tau = t / b
    bi = h * b / k
    lam = np.concatenate(([0.0], j1_zeros(nterms)))[1:]
    flux = q_w / (math.pi * a * a)
    qn = 2.0 * flux * eps * j1(lam * eps) / (lam * j0(lam) ** 2)       # Fourier-Bessel coefficients of the flux
    th = np.tanh(lam * tau)
    # A_n Z_n(0) with the overflow-safe tanh form
    an_z0 = (qn * b / k) * (1.0 + (bi / lam) * th) / (lam * th + bi)
    mean_weight = 2.0 * j1(lam * eps) / (lam * eps)
    spread_mean = float(np.sum(an_z0 * mean_weight))
    spread_centre = float(np.sum(an_z0))
    oned_joint = q_w / (math.pi * b * b * h)
    oned_plate = q_w * t / (k * math.pi * b * b)
    return {
        "mean": oned_joint + oned_plate + spread_mean,
        "centre": oned_joint + oned_plate + spread_centre,
        "oned_plate": oned_plate,
        "oned_joint": oned_joint,
        "spreading_mean": spread_mean,
    }


# ---------------------------------------------------------------- independent finite-volume solver
def _graded_widths(length, n, ratio):
    """n cell widths summing to length, each ratio times the previous (first is smallest if ratio > 1)."""
    if ratio == 1.0:
        return np.full(n, length / n)
    w0 = length * (ratio - 1.0) / (ratio ** n - 1.0)
    return w0 * ratio ** np.arange(n)


def spreading_fv(q_w, a, b, t, k, h, n_in=80, n_out=80, nz=60, r_ratio=1.06, z_ratio=1.09):
    """Axisymmetric finite-volume solve of the same problem. Returns (mean source-surface T, total flux to fluid)."""
    if not (0 < a < b) or t <= 0 or k <= 0 or h <= 0 or q_w <= 0:
        raise ValueError("bad spreading inputs (finite-volume needs a < b)")
    w_in = _graded_widths(a, n_in, r_ratio)[::-1]          # finest next to r = a
    w_out = _graded_widths(b - a, n_out, r_ratio)          # finest next to r = a
    wr = np.concatenate((w_in, w_out))
    rf = np.concatenate(([0.0], np.cumsum(wr)))            # radial faces
    rc = 0.5 * (rf[:-1] + rf[1:])
    wz = _graded_widths(t, nz, z_ratio)                    # finest at the source face
    nr = len(wr)
    ann = math.pi * (rf[1:] ** 2 - rf[:-1] ** 2)           # annulus areas (top/bottom of each cell)
    flux = q_w / (math.pi * a * a)
    in_src = rf[1:] <= a * (1 + 1e-12)

    def idx(i, j):
        return i * nz + j

    rows, cols, vals = [], [], []
    rhs = np.zeros(nr * nz)
    diag = np.zeros(nr * nz)

    def link(p, q_, g):
        rows.extend((p, q_))
        cols.extend((q_, p))
        vals.extend((-g, -g))
        diag[p] += g
        diag[q_] += g

    for i in range(nr):
        for j in range(nz):
            p = idx(i, j)
            if i + 1 < nr:                                   # radial neighbour
                g = k * 2 * math.pi * rf[i + 1] * wz[j] / (rc[i + 1] - rc[i])
                link(p, idx(i + 1, j), g)
            if j + 1 < nz:                                   # axial neighbour
                dzc = 0.5 * (wz[j] + wz[j + 1])
                link(p, idx(i, j + 1), k * ann[i] / dzc)
            if j == 0 and in_src[i]:                         # source flux
                rhs[p] += flux * ann[i]
            if j == nz - 1:                                  # bottom: half-cell conduction in series with h
                g = 1.0 / (0.5 * wz[j] / (k * ann[i]) + 1.0 / (h * ann[i]))
                diag[p] += g
    rows.extend(range(nr * nz))
    cols.extend(range(nr * nz))
    vals.extend(diag)
    mat = coo_matrix((vals, (rows, cols)), shape=(nr * nz, nr * nz)).tocsr()
    temp = spsolve(mat, rhs)
    t_top = np.array([temp[idx(i, 0)] for i in range(nr)])
    # source-surface temperature = cell centre + half-cell conduction drop
    t_surf = t_top + flux * 0.5 * wz[0] / k
    mean = float(np.sum(t_surf[in_src] * ann[in_src]) / np.sum(ann[in_src]))
    out = 0.0
    for i in range(nr):
        g = 1.0 / (0.5 * wz[-1] / (k * ann[i]) + 1.0 / (h * ann[i]))
        out += g * temp[idx(i, nz - 1)]
    return mean, float(out)


# ---------------------------------------------------------------- closed-form thin-plate (fin) limit
def fin_closed_form_mean(q_w, a, k, t, h):
    """Mean source-disk temperature rise for a thin plate (no through-thickness gradient) of infinite radius.

    Plate equation  k t lap(T) - h T + q 1[r<a] = 0  gives
        T_mean = (q/h) * (1 - 2 K1(m a) I1(m a)),   m = sqrt(h / (k t)),  q = Q / (pi a^2).
    Derived independently of the eigenfunction series; used as a regime check.
    """
    from scipy.special import i1, k1
    m = math.sqrt(h / (k * t))
    q = q_w / (math.pi * a * a)
    return (q / h) * (1.0 - 2.0 * k1(m * a) * i1(m * a))


def area_multiple(d_total_k, t_j_k=bud.T_J_BASE, **kw):
    """Radiator area relative to a zero-drop stack at the same die temperature t_j_k.

    Reads: the mean die surface is held at t_j_k, the panel sits at t_j_k - d_total_k, q_abs as given.
    Returns None only when the panel cannot reject net heat at that temperature (absorbed flux exceeds
    emission). Any bad input still raises.
    """
    t_rad = t_j_k - d_total_k
    if t_rad <= 0:
        raise ValueError("panel temperature must be positive kelvin")
    eps = kw.get("eps", bud.EPS)
    if eps * bud.SIGMA * t_rad ** 4 - kw.get("q_abs", bud.Q_ABS) <= 0:
        return None
    return bud.radiator_area(Q_W, t_rad, **kw) / bud.radiator_area(Q_W, t_j_k, **kw)


def fin_length_mm(kt_w_per_k, h=H_JOINT):
    """Fin length sqrt(k t / h) in mm."""
    return math.sqrt(kt_w_per_k / h) * 1e3


def keff_from_kt(kt_w_per_k, t_m):
    return kt_w_per_k / t_m


def mult_str(m):
    return "infeasible" if m is None else f"{m:.2f}x"


# ---------------------------------------------------------------- whole stack
def stack_drops(q_w, die_area_m2, footprint_m2, t, k, tim_cm2k_per_w, h=H_JOINT):
    """Drops (K) from die surface to the panel: TIM, plate 1D, spreading, joint. Mean over the die."""
    a = circle_radius(die_area_m2)
    b = circle_radius(footprint_m2)
    s = spreading_series(q_w, a, b, t, k, h)
    d_tim = bud.drop_area_specific(q_w, die_area_m2, bud.cm2k_per_w_to_m2k_per_w(tim_cm2k_per_w))
    return {"tim": d_tim, "plate_1d": s["oned_plate"], "spreading": s["spreading_mean"],
            "joint": s["oned_joint"], "total": d_tim + s["mean"], "peak_extra": s["centre"] - s["mean"]}


# ---------------------------------------------------------------- self-checks
def close(a, b, tol):
    return abs(a - b) <= tol * max(abs(b), 1e-30)


def _buggy_centre_not_mean(q_w, a, b, t, k, h):
    return spreading_series(q_w, a, b, t, k, h)["centre"]          # deliberately the wrong average


def _buggy_no_bi(q_w, a, b, t, k, h):
    """Series with the Biot term dropped from Z_n(0): treats the bottom as isothermal in the mode shapes."""
    eps, tau = a / b, t / b
    lam = j1_zeros(N_SERIES)
    flux = q_w / (math.pi * a * a)
    qn = 2.0 * flux * eps * j1(lam * eps) / (lam * j0(lam) ** 2)
    th = np.tanh(lam * tau)
    an_z0 = (qn * b / k) / (lam * th)
    spread = float(np.sum(an_z0 * 2.0 * j1(lam * eps) / (lam * eps)))
    return q_w / (math.pi * b * b * h) + q_w * t / (k * math.pi * b * b) + spread


def run_self_checks():
    results = []

    def record(name, cond):
        results.append((name, bool(cond)))

    # (1) boundary: source covering the whole plate has no spreading at all
    s_full = spreading_series(500.0, 0.05, 0.05, 0.01, 200.0, 3000.0)
    record("a = b: spreading term is zero", abs(s_full["spreading_mean"]) < 1e-9 * s_full["mean"])
    record("a = b: total equals the 1D plate plus joint",
           close(s_full["mean"], 500.0 / (math.pi * 0.05 ** 2) * (0.01 / 200.0 + 1 / 3000.0), 1e-9))

    # (2) independent cross-check: series vs finite-volume, deliberately unequal cases
    cases = [
        ("thin plate, weak joint", dict(q_w=700.0, a=0.0161, b=0.113, t=0.002, k=167.0, h=4030.0)),
        ("thick plate, strong joint", dict(q_w=300.0, a=0.02, b=0.09, t=0.015, k=390.0, h=20000.0)),
        ("small footprint ratio, high k", dict(q_w=700.0, a=0.0161, b=0.04, t=0.005, k=1000.0, h=4030.0)),
    ]
    for name, c in cases:
        ser = spreading_series(**c)["mean"]
        fv, flux_out = spreading_fv(**c)
        record(f"series matches finite-volume within 0.5%: {name}", close(fv, ser, 5e-3))
        record(f"finite-volume conserves energy to 1e-6: {name}", close(flux_out, c["q_w"], 1e-6))
    # grid convergence of the finite-volume solver
    c = cases[0][1]
    coarse, _ = spreading_fv(**c)
    fine, _ = spreading_fv(**c, n_in=160, n_out=160, nz=120, r_ratio=1.03, z_ratio=1.045)
    record("finite-volume result changes under 0.3% on a 2x refined grid", close(coarse, fine, 3e-3))

    # (3) half-space limit: excess over the 1D part tends to 8/(3 pi^2) Q/(k a) for a uniform-flux disk
    a, k = 0.01, 200.0
    s_hs = spreading_series(100.0, a, 100 * a, 100 * a, k, 1e6, nterms=6000)
    target = 8.0 / (3.0 * math.pi ** 2) * 100.0 / (k * a)
    record("half-space limit within 3% of 8/(3 pi^2) Q/(k a)", close(s_hs["spreading_mean"], target, 3e-2))

    # (3b) thin-plate regime: closed-form I/K Bessel fin solution vs the eigenfunction series
    for (kk, tt) in ((390.0, 0.0005), (167.0, 0.0003)):
        fin = fin_closed_form_mean(700.0, 0.0161, kk, tt, 4030.0)
        ser_thin = spreading_series(700.0, 0.0161, 0.6, tt, kk, 4030.0, nterms=6000)["mean"]
        record(f"thin-plate closed form matches the series within 2%: k={kk:.0f}, t={tt * 1e3:.1f} mm", close(ser_thin, fin, 2e-2))
    no_spreading = (700.0 / (math.pi * 0.0161 ** 2)) / 4030.0          # joint under the die flux, plate does nothing
    fin_ref = fin_closed_form_mean(700.0, 0.0161, 390.0, 0.0005, 4030.0)
    record("closed-form check fires on the no-spreading value q/h (a plate that does nothing)", not close(no_spreading, fin_ref, 2e-2))
    record("spreading always beats no spreading: fin result is below q/h and above zero", 0 < fin_ref < no_spreading)

    # (4) the cross-check gate must fire on wrong models
    c = cases[0][1]
    fv_ref, _ = spreading_fv(**c)
    record("gate fires on centre-instead-of-mean", not close(_buggy_centre_not_mean(**c), fv_ref, 5e-3))
    record("gate fires on a series with the Biot term dropped", not close(_buggy_no_bi(**c), fv_ref, 5e-3))

    # (5) physical ordering
    base = spreading_series(700.0, 0.0161, 0.113, 0.005, 167.0, 4030.0)["spreading_mean"]
    record("higher conductivity lowers spreading excess",
           spreading_series(700.0, 0.0161, 0.113, 0.005, 390.0, 4030.0)["spreading_mean"] < base)
    record("smaller die raises spreading excess at the same power",
           spreading_series(700.0, 0.008, 0.113, 0.005, 167.0, 4030.0)["spreading_mean"] > base)
    record("peak is above mean", spreading_series(700.0, 0.0161, 0.113, 0.005, 167.0, 4030.0)["centre"]
           > spreading_series(700.0, 0.0161, 0.113, 0.005, 167.0, 4030.0)["mean"])
    record("series term count 4000 vs 8000 agrees to 1e-6",
           close(spreading_series(700.0, 0.0161, 0.113, 0.002, 167.0, 4030.0, nterms=4000)["mean"],
                 spreading_series(700.0, 0.0161, 0.113, 0.002, 167.0, 4030.0, nterms=8000)["mean"], 1e-6))

    # (6) stack bookkeeping: the pieces sum to the total
    d = stack_drops(700.0, DIE_AREA_M2, FOOTPRINT_M2, 0.005, 390.0, 0.039)
    record("stack pieces sum to the total", close(d["tim"] + d["plate_1d"] + d["spreading"] + d["joint"], d["total"], 1e-12))
    record("stack joint term equals the 400 cm2 joint of the first script",
           close(d["joint"], bud.drop_joint(700.0, FOOTPRINT_M2), 1e-12))

    # (6b) hand-computed known answers for the area multiple, stack pieces and table helpers
    record("area multiple of a zero drop is 1", close(area_multiple(0.0), 1.0, 1e-12))
    record("area multiple of 10 K at q_abs = 0 equals (333.15/323.15)^4",
           close(area_multiple(10.0), (333.15 / 323.15) ** 4, 1e-12))
    record("area multiple is above 1 for a positive drop", area_multiple(5.0) > 1.0)
    record("area multiple returns None when absorbed flux exceeds emission", area_multiple(90.0, q_abs=400.0) is None)
    for label, call in [("area multiple raises on a drop above the die temperature", lambda: area_multiple(400.0)),
                        ("area multiple raises on emissivity above 1", lambda: area_multiple(5.0, eps=1.5))]:
        try:
            call()
            record(label, False)
        except ValueError:
            record(label, True)
    d1 = stack_drops(700.0, DIE_AREA_M2, FOOTPRINT_M2, 0.005, 390.0, 0.039)
    d2 = stack_drops(700.0, DIE_AREA_M2, FOOTPRINT_M2, 0.010, 390.0, 0.039)
    record("TIM drop uses the die area: 700/814e-6 * 3.9e-6", close(d1["tim"], 700.0 / 814e-6 * 3.9e-6, 1e-12))
    record("plate 1D term doubles when thickness doubles", close(d2["plate_1d"], 2 * d1["plate_1d"], 1e-12))
    record("plate 1D term equals Q t / (k pi b^2) by hand",
           close(d1["plate_1d"], 700.0 * 0.005 / (390.0 * FOOTPRINT_M2), 1e-9))
    record("peak extra is positive", d1["peak_extra"] > 0)
    record("material table holds Cu 390 and Al 167 W/m/K", dict(K_MATERIALS)["Cu"] == 390.0 and dict(K_MATERIALS)["Al 6061"] == 167.0)
    record("fin length of Cu 5 mm (k t = 1.95 W/K) at h = 4030 is 22.0 mm", close(fin_length_mm(1.95), 22.0, 5e-3))
    record("k_eff from k t = 50 W/K at 5 mm is 10000", close(keff_from_kt(50.0, 5e-3), 10000.0, 1e-12))
    record("k t that makes the fin length equal the footprint radius is h b^2",
           close(fin_length_mm(H_JOINT * circle_radius(FOOTPRINT_M2) ** 2), circle_radius(FOOTPRINT_M2) * 1e3, 1e-12))

    # (7) invalid inputs raise
    for label, call in [
        ("die larger than footprint raises", lambda: spreading_series(700.0, 0.2, 0.1, 0.005, 100.0, 4000.0)),
        ("zero conductivity raises", lambda: spreading_series(700.0, 0.01, 0.1, 0.005, 0.0, 4000.0)),
        ("zero thickness raises", lambda: spreading_series(700.0, 0.01, 0.1, 0.0, 100.0, 4000.0)),
        ("finite-volume rejects a = b", lambda: spreading_fv(700.0, 0.1, 0.1, 0.005, 100.0, 4000.0)),
        ("negative area raises", lambda: circle_radius(-1.0)),
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

    a = circle_radius(DIE_AREA_M2)
    b = circle_radius(FOOTPRINT_M2)
    tim = bud.SOLDER_TIM_CM2K_PER_W["52In/48Sn"]
    print("INPUTS")
    print(f"  Q = {Q_W:.0f} W; die {DIE_AREA_M2 * 1e6:.0f} mm2 -> equal-area circle radius {a * 1e3:.2f} mm")
    print(f"  footprint {FOOTPRINT_M2 * 1e4:.0f} cm2 -> equal-area circle radius {b * 1e3:.1f} mm (area ratio {FOOTPRINT_M2 / DIE_AREA_M2:.0f}x)")
    print(f"  joint h = {H_JOINT:.0f} W/m2/K; solder TIM 52In/48Sn {tim} cm2 C/W; materials are handbook k (sources in the file), diamond-class ASSUMED")
    print("  SCOPE: uniform die flux, periodic-array edge, no silicon / contact resistance. Totals are still lower bounds.")

    print("\nTABLE A  die-surface (mean) to panel, footprint 400 cm2, solder TIM 52In/48Sn. Drops in K.")
    print("  'area x' = radiator area vs a zero-drop stack when the MEAN (or uniform-flux PEAK) die surface is held at 60 C, q_abs = 0.")
    print("  Multiples are not additive across layers (T^4 law). The zero-drop baseline is unreachable. No silicon resistance is included.")
    print("  plate           | t_mm | TIM   | plate 1D | spreading | joint | total  | peak extra | area x mean | area x peak")
    for name, k in K_MATERIALS:
        for t in T_PLATE_M:
            d = stack_drops(Q_W, DIE_AREA_M2, FOOTPRINT_M2, t, k, tim)
            print(f"  {name:14s}  | {t * 1e3:4.0f} | {d['tim']:5.2f} | {d['plate_1d']:8.2f} | {d['spreading']:9.2f} | "
                  f"{d['joint']:5.2f} | {d['total']:6.2f} | {d['peak_extra']:10.2f} | {mult_str(area_multiple(d['total'])):>11s} | {mult_str(area_multiple(d['total'] + d['peak_extra'])):>11s}")

    print("\nTABLE B  footprint sweep, Cu 5 mm, solder TIM 52In/48Sn. All values K.")
    print("  footprint_cm2 | spreading | joint | plate 1D | total | area x mean")
    for fp in FOOTPRINTS_CM2:
        d = stack_drops(Q_W, DIE_AREA_M2, fp * 1e-4, 5e-3, 390.0, tim)
        print(f"  {fp:12d}  | {d['spreading']:9.2f} | {d['joint']:5.2f} | {d['plate_1d']:8.2f} | {d['total']:5.2f} | {mult_str(area_multiple(d['total'])):>10s}")

    print("\nTABLE C  full worked stacks (TIM + plate + spreading + joint at 400 cm2), plate 5 mm.")
    print("  Radiator area multiple if the die surface is held at 60 C (T_rad = 60 C minus the total), one side.")
    print("  plate          | total K | T_rad C | area x, q_abs=0 | q_abs=100  | q_abs=200 | spreading alone x (q_abs=0)")
    for name, k in K_MATERIALS:
        d = stack_drops(Q_W, DIE_AREA_M2, FOOTPRINT_M2, 5e-3, k, tim)
        cells = [mult_str(area_multiple(d["total"], q_abs=qa)) for qa in bud.Q_ABS_SWEEP]
        print(f"  {name:14s} | {d['total']:7.2f} | {bud.T_J_BASE - d['total'] - 273.15:7.1f} | {cells[0]:>15s} | {cells[1]:>10s} | {cells[2]:>10s} | {mult_str(area_multiple(d['spreading'])):>14s}")

    print("\nTABLE D  how much spreader conductance-thickness (k*t) the footprint needs. Isotropic effective k, t = 5 mm, footprint 400 cm2.")
    kt_need = H_JOINT * b * b
    print(f"  fin length sqrt(k t / h) must approach the footprint radius {b * 1e3:.0f} mm, i.e. k*t of about h*b^2 = {kt_need:.0f} W/K")
    print("  k*t W/K | k_eff W/m/K | fin length mm | spreading K | total K (with TIM, joint) | area x (q_abs=0)")
    for kt in (2.0, 5.0, 15.0, 50.0, 150.0, 500.0):
        keff = keff_from_kt(kt, 5e-3)
        d = stack_drops(Q_W, DIE_AREA_M2, FOOTPRINT_M2, 5e-3, keff, tim)
        lf = fin_length_mm(kt)
        print(f"  {kt:7.0f} | {keff:11.0f} | {lf:13.0f} | {d['spreading']:11.2f} | {d['total']:25.2f} | {mult_str(area_multiple(d['total'])):>16s}")
    print("  Reference k*t: Cu 5 mm = 1.95 W/K, diamond-class 5 mm = 5 W/K. An effective figure for an embedded heat-pipe panel is not")
    print("  taken from any source here; a real panel is anisotropic, so rows at high k*t are an idealised bound.")

    print("\nSELF-CHECKS")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  {len(checks)}/{len(checks)} passed")
    print("  Table rows are not individually tested; the pieces they use (area multiple, stack pieces, fin length, k_eff) have known-answer tests.")
    print("  Verification stops here: the series solution is cross-checked against an independent finite-volume solve of the same")
    print("  idealised geometry. Neither says anything about a real, square, non-uniformly heated die on a shared panel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
