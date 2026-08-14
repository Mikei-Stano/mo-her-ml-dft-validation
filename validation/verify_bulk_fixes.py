"""Verification of the bulk-structure fixes: MEASURED values against CLAIMED ones.

MoB serves as the calibration point: it was not modified and must reproduce
1.8493 Å / 8.601 g/cm3. If any fix fails to reproduce its claimed number the
run reports FAIL and that structure must not go into production.
"""
import sys
import types

import numpy as np

sys.path.insert(0, "scripts")

# generate_structures imports things that may be absent in a clean environment
for missing in ("gpaw", "gpaw.mpi"):
    if missing not in sys.modules:
        try:
            __import__(missing)
        except ImportError:
            sys.modules[missing] = types.ModuleType(missing)

import generate_structures as g  # noqa: E402

# (function, claimed min-dist [A], claimed density [g/cm3], claimed atom count, density tolerance %)
CASES = [
    ("create_mob_bulk",  1.8493, 8.601,  8,  0.5, "CALIBRATION - unchanged"),
    ("create_mop_bulk",  2.4495, 7.358,  2,  0.5, "fixed: WC type P-6m2"),
    ("create_mos2_bulk", 2.418,  4.997,  6,  1.0, "fixed: 2H P6_3/mmc"),
    ("create_mose2_bulk", 2.4907, 6.9814, 6,  1.0, "fixed: 2H P6_3/mmc"),
    ("create_mo2c_bulk", 2.096,  9.185, 12,  1.0, "fixed: Pbcn expanded"),
    ("create_mo2n_bulk", 2.0808, 9.4880, 6,  1.0, "fixed: Fm-3m defect rock salt"),
    ("create_ti3c2_bulk", 1.993, None,   7,  None, "fixed: P-3m1, density N/A (monolayer)"),
]

AMU = 1.66053906660e-24  # g


def _min_same(atoms, sym):
    """Smallest distance between atoms of the same element (PBC-aware)."""
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    idx = [i for i, a in enumerate(atoms) if a.symbol == sym]
    return float(min(d[i][j] for i in idx for j in idx if i != j))


def measure(atoms):
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    mind = float(d.min())
    vol = float(atoms.get_volume())              # Å³
    mass = float(sum(atoms.get_masses()))        # amu
    dens = mass * AMU / (vol * 1e-24)            # g/cm³
    return len(atoms), mind, dens, vol, atoms.get_chemical_formula()


print("%-22s %5s %-12s %10s %10s %8s  %s"
      % ("function", "at.", "formula", "min-dist", "density", "verdict", "note"))
print("-" * 108)
fails = []
for fn, exp_d, exp_rho, exp_n, tol, note in CASES:
    try:
        a = getattr(g, fn)()
        n, mind, rho, vol, formula = measure(a)
    except Exception as exc:
        print("%-22s  CHYBA: %s" % (fn, exc))
        fails.append((fn, "exception: %s" % exc)); continue

    problems = []
    if n != exp_n:
        problems.append("atoms %d != %d" % (n, exp_n))
    if abs(mind - exp_d) > 0.01:
        problems.append("min-dist %.4f != %.4f" % (mind, exp_d))
    if exp_rho is not None and abs(rho - exp_rho) / exp_rho * 100 > tol:
        problems.append("density %.4f vs claimed %.4f (%.2f %%)"
                        % (rho, exp_rho, abs(rho - exp_rho) / exp_rho * 100))

    verdikt = "OK" if not problems else "FAIL"
    if problems:
        fails.append((fn, "; ".join(problems)))
    print("%-22s %5d %-12s %10.4f %10.4f %8s  %s"
          % (fn, n, formula, mind, rho, verdikt, note))

print()
if fails:
    print("!!! FAILED %d of %d - must not go into production:" % (len(fails), len(CASES)))
    for fn, why in fails:
        print("   %-22s %s" % (fn, why))
    sys.exit(1)
print("All %d bulk structures passed verification (including the MoB calibration)." % len(CASES))

# -- check the AA stacking-fault guard in create_slab -----------------------
# NOTE: `miller` is a STRING ("(111)"), not a tuple - create_slab calls _parse_miller.
print("\n=== create_slab: AA stacking-fault guard ===")
print("referencia: v opravenom γ-Mo2N bulku je Mo–Mo = 2.9427 Å")
slab_fail = []
for facet in ("(100)", "(110)", "(111)"):
    for L in (3, 4, 5, 6):
        try:
            s = g.create_slab(g.create_mo2n_bulk(), miller=facet,
                              size=(2, 2, L), vacuum=8)
            got = _min_same(s, "Mo")
            ok = got >= 2.7
            print("  %s layers=%d  n=%3d  min Mo–Mo = %.4f Å  %s"
                  % (facet, L, len(s), got, "OK" if ok else "FAIL"))
            if not ok:
                slab_fail.append((facet, L, got))
        except Exception as exc:
            print("  %s layers=%d  CHYBA: %s" % (facet, L, exc))
            slab_fail.append((facet, L, str(exc)))

print()
if slab_fail:
    print("!!! the guard MISSED a stacking fault in %d cases:" % len(slab_fail))
    for f, L, v in slab_fail:
        print("   %s layers=%d → %s" % (f, L, v))
    sys.exit(1)
print("The stacking-fault guard works: every cut and layer count gives a correct "
      "Mo-Mo distance (previously (111)/layers=4 gave 2.4027 A).")
