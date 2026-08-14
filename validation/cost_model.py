"""Cost of the remaining work, from MEASURED single-points rather than extrapolated old logs.

CALIBRATION POINTS (h=0.16, 4x4x1 mesh, a single-point PAIR of clean+adslab on
a fixed UMA geometry - exactly the production physics, only without relaxation):
    MoP_(111)     30 atoms    750 s on 16 ranks  =    3.33 core-h / pair
    Mo2N_(100)    96 atoms  15803 s on 32 ranks  =  140.47 core-h / pair
The implied scaling exponent is ln(140.47/3.33) / ln(96/30) = 3.2, consistent
with LCAO being dominated by an O(N_basis^3) diagonalisation per k-point (GPAW
timer: Orbital Layouts 21.6 % + DenseAtomicCorrection 20.5 % + an h^-3 part
at 55.8 %).

THE RELAXATION FACTOR is the largest uncertainty and is labelled as such. It
combines:
  - the number of ionic steps: over an uncensored subset of 128 BFGS logs the
    Kaplan-Meier median at fmax=0.05 is 29 steps for the clean side, while the
    adslab curve never reaches 50 %, so its median exceeds 30. A budget must be
    built on p90-p95, not on the median.
  - the cost of a step relative to a FRESH single-point: each further step
    starts from the previous density, so the SCF converges in fewer iterations.
    Without a measurement at h=0.16
    beriem 0.40-0.55.
"""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# -- calibration -------------------------------------------------------------
CAL = [(30, 750 * 16 / 3600), (96, 15803 * 32 / 3600)]   # (atoms, core-h/pair)
import math
EXP = math.log(CAL[1][1] / CAL[0][1]) / math.log(CAL[1][0] / CAL[0][0])
A = CAL[0][1] / CAL[0][0] ** EXP


def sp_pair(n):
    """core-h for one single-point pair (clean+adslab) at h=0.16, 4x4x1."""
    return A * n ** EXP


# -- campaign structures -----------------------------------------------------
rows = list(csv.DictReader(open(REPO / "data/adsorbml_results/ranked_candidates_campaign.csv")))
CONVERGED = {"MoP", "MoS2", "MoSe2", "Ti3C2O2"}      # 4x4x1 demonstrably suffices
METALLIC = {"MoB", "Mo2C", "Mo2N"}                    # need 6x6x1 (n_IBZ 8->18)
K_PENALTY = 18 / 8                                    # 2.25x more k-point work
K_SHARE = 0.62                                        # measured k-dependent share

# 6x6x1 does not make the whole calculation 2.25x more expensive, only its k-dependent part
METAL_FACTOR = (1 - K_SHARE) + K_SHARE * K_PENALTY

STEPS = {"optimisticky": (24, 0.40), "realne": (40, 0.48), "pesimisticky": (60, 0.55)}

print("=" * 78)
print(f"CALIBRATION: core-h/pair = {A:.4g} * N^{EXP:.2f}     (2 measured points)")
print(f"             metallic families x {METAL_FACTOR:.2f} for the 6x6x1 mesh")
print("=" * 78)

groups = {"A (converged, 4x4x1)": CONVERGED, "B (metallic, 6x6x1)": METALLIC}
sp_tot = {}
for label, fams in groups.items():
    sel = [r for r in rows if r["family"] in fams]
    fac = METAL_FACTOR if fams is METALLIC else 1.0
    tot = sum(sp_pair(int(r["n_atoms_slab"])) * fac for r in sel)
    sp_tot[label] = tot
    ns = sorted(int(r["n_atoms_slab"]) for r in sel)
    print(f"\n{label}:  {len(sel)} structures, {ns[0]}-{ns[-1]} atoms (median {ns[len(ns)//2]})")
    print(f"  sum of single-point pairs: {tot:8.0f} core-h")
    for r in sorted(sel, key=lambda r: -sp_pair(int(r['n_atoms_slab'])))[:3]:
        n = int(r["n_atoms_slab"])
        print(f"    most expensive: {r['slab_name']:<24} {n:3d} at.  {sp_pair(n)*fac:8.0f} core-h/pair")

print("\n" + "=" * 78)
print("FULL RELAXATION (both sides), by step budget - AN ESTIMATE, not a measurement")
print("=" * 78)
print(f"  {'scenario':<14}{'steps/side':>14}{'cost/step':>11}"
      f"{'campaign A':>12}{'campaign B':>12}{'TOTAL':>12}")
for name, (steps, frac) in STEPS.items():
    mult = steps * frac
    a = sp_tot["A (converged, 4x4x1)"] * mult
    b = sp_tot["B (metallic, 6x6x1)"] * mult
    print(f"  {name:<14}{steps:>14}{frac:>11.2f}{a:>12.0f}{b:>12.0f}{a+b:>12.0f}")

print("\n" + "=" * 78)
FAT_LEFT = 513597
print(f"ZOSTATOK FAT: {FAT_LEFT} core·h")
for name, (steps, frac) in STEPS.items():
    mult = steps * frac
    t = (sp_tot["A (converged, 4x4x1)"] + sp_tot["B (metallic, 6x6x1)"]) * mult
    print(f"  {name:<14} {t:8.0f} core·h = {100*t/FAT_LEFT:5.1f} % zostatku,"
          f"  rezerva {FAT_LEFT/t:.1f}×")
print("=" * 78)
