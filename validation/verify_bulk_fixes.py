"""Overenie opráv bulk štruktúr — porovná NAMERANÉ hodnoty s TVRDENÝMI.

MoB slúži ako kalibračný bod (nemenil sa, musí dať 1.8493 Å / 8.601 g/cm³).
Ak niektorá oprava nedá tvrdené číslo, vypíše sa FAIL a nesmie ísť do produkcie.
"""
import sys
import types

import numpy as np

sys.path.insert(0, "scripts")

# generate_structures importuje veci, ktoré v čistom prostredí nemusia byť
for missing in ("gpaw", "gpaw.mpi"):
    if missing not in sys.modules:
        try:
            __import__(missing)
        except ImportError:
            sys.modules[missing] = types.ModuleType(missing)

import generate_structures as g  # noqa: E402

# (funkcia, tvrdená min-dist [Å], tvrdená hustota [g/cm³], tvrdený počet atómov, tolerancia hustoty %)
CASES = [
    ("create_mob_bulk",  1.8493, 8.601,  8,  0.5, "KALIBRÁCIA — nemenené"),
    ("create_mop_bulk",  2.4495, 7.358,  2,  0.5, "opravené: WC-typ P-6m2"),
    ("create_mos2_bulk", 2.418,  4.997,  6,  1.0, "opravené: 2H P6_3/mmc"),
    ("create_mose2_bulk", 2.4907, 6.9814, 6,  1.0, "opravené: 2H P6_3/mmc"),
    ("create_mo2c_bulk", 2.096,  9.185, 12,  1.0, "opravené: Pbcn expandované"),
    ("create_mo2n_bulk", 2.0808, 9.4880, 6,  1.0, "opravené: Fm-3m defect rock-salt"),
    ("create_ti3c2_bulk", 1.993, None,   7,  None, "opravené: P-3m1, hustota N/A (monovrstva)"),
]

AMU = 1.66053906660e-24  # g


def _min_same(atoms, sym):
    """Min. vzdialenosť medzi atómami toho istého prvku (PBC-aware)."""
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
      % ("funkcia", "at.", "formula", "min-dist", "hustota", "verdikt", "poznámka"))
print("-" * 108)
fails = []
for fn, exp_d, exp_rho, exp_n, tol, note in CASES:
    try:
        a = getattr(g, fn)()
        n, mind, rho, vol, formula = measure(a)
    except Exception as exc:
        print("%-22s  CHYBA: %s" % (fn, exc))
        fails.append((fn, "výnimka: %s" % exc)); continue

    problems = []
    if n != exp_n:
        problems.append("atómov %d != %d" % (n, exp_n))
    if abs(mind - exp_d) > 0.01:
        problems.append("min-dist %.4f != %.4f" % (mind, exp_d))
    if exp_rho is not None and abs(rho - exp_rho) / exp_rho * 100 > tol:
        problems.append("hustota %.4f vs tvrdená %.4f (%.2f %%)"
                        % (rho, exp_rho, abs(rho - exp_rho) / exp_rho * 100))

    verdikt = "OK" if not problems else "FAIL"
    if problems:
        fails.append((fn, "; ".join(problems)))
    print("%-22s %5d %-12s %10.4f %10.4f %8s  %s"
          % (fn, n, formula, mind, rho, verdikt, note))

print()
if fails:
    print("!!! NEPREŠLO %d z %d — do produkcie NESMIE:" % (len(fails), len(CASES)))
    for fn, why in fails:
        print("   %-22s %s" % (fn, why))
    sys.exit(1)
print("VŠETKÝCH %d bulk štruktúr prešlo overením (vrátane kalibračného MoB)." % len(CASES))

# ── kontrola brány proti AA stacking faultu v create_slab ─────────────────
# POZOR: `miller` je STRING ("(111)"), nie tuple — create_slab volá _parse_miller.
print("\n=== create_slab: brána proti AA stacking faultu ===")
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
    print("!!! brána NEZACHYTILA stacking fault v %d prípadoch:" % len(slab_fail))
    for f, L, v in slab_fail:
        print("   %s layers=%d → %s" % (f, L, v))
    sys.exit(1)
print("Brána proti stacking faultu funguje — všetky rezy a počty vrstiev dávajú "
      "korektnú Mo–Mo vzdialenosť (predtým (111)/layers=4 dávalo 2.4027 Å).")
