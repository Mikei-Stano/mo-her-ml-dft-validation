"""
ANOMALY RECLASSIFICATION - removing false positives caused by a single symmetric threshold.

PROBLEM (diagnosed): the first detector applied ONE threshold (cushion 1.2)
symmetrically. A bond sitting exactly on it flips under any relaxation, so it is
flagged for every candidate. VERIFIED on MoB_(111)_vac2B: Mo-B goes 2.885 ->
2.797 A against a threshold of 2.856, a change of only 0.064-0.088 A, and the
sensitivity is inconsistent - c=1.2 flags 5/5 while c=1.3 flags 2/5. That is an
artefact, not physics.

SOLUTION - HYSTERESIS, as the canonical ocdata `DetectTrajAnomaly` does it:
  formation = bound in the adslab at mult=BOND (1.0) AND unbound in the clean
              slab at mult=TOL (1.3)
  breaking  = bound in the clean slab at mult=BOND (1.0) AND unbound in the
              adslab at mult=TOL (1.3)
Between 1.0 and 1.3 lies a dead zone where nothing is flagged, so borderline
bonds stop flickering. In addition the distance must genuinely change by at
least MIN_DELTA (0.2 A): a bond cannot "form" through a 0.08 A displacement.

Runs on CPU over the already computed candidate_*.traj files - it needs no GPU
and no new calculation. It rewrites the anomaly columns in candidates.csv and
keeps the originals for comparison.
"""
import csv
import glob
import os
import sys

import numpy as np
from ase.io import read
from ase.data import covalent_radii
from ase.geometry import find_mic

RES = "data/adsorbml_results_campaign"
CLEAN = "data/uma_relaxed_campaign"
MULT_BOND, MULT_TOL = 1.0, 1.3     # hysteresis band
MIN_DELTA = 0.2                    # A - minimum genuine change in distance
MAX_SLAB_MOVE = 1.5                # Å


def bonded(atoms, mult):
    d = atoms.get_all_distances(mic=True)
    r = np.array([covalent_radii[n] for n in atoms.get_atomic_numbers()])
    thr = mult * (r[:, None] + r[None, :])
    np.fill_diagonal(d, np.inf)
    return (d < thr), d


def classify(clean, adslab):
    """Anomalies with hysteresis. Returns (flags, diag)."""
    ns = len(clean)
    if len(adslab) != ns + 1:
        return ["atom_count_mismatch"], {}
    sub = adslab[:ns]

    b0_hard, d0 = bonded(clean, MULT_BOND)
    b0_soft, _ = bonded(clean, MULT_TOL)
    b1_hard, d1 = bonded(sub, MULT_BOND)
    b1_soft, _ = bonded(sub, MULT_TOL)

    # formation: firmly bound in the adslab AND not even loosely bound in the clean slab
    formed = np.triu(b1_hard & ~b0_soft, 1)
    broken = np.triu(b0_hard & ~b1_soft, 1)

    # + the requirement of a genuine change in distance
    delta = np.abs(d1 - d0)
    formed &= (delta >= MIN_DELTA)
    broken &= (delta >= MIN_DELTA)

    flags = []
    nf, nb = int(formed.sum()), int(broken.sum())
    if nf:
        flags.append(f"surface_bonds_formed:{nf}")
    if nb:
        flags.append(f"surface_bonds_broken:{nb}")

    dd, _ = find_mic(sub.get_positions() - clean.get_positions(), clean.cell, pbc=True)
    maxmove = float(np.linalg.norm(dd, axis=1).max())
    if maxmove > MAX_SLAB_MOVE:
        flags.append(f"slab_atom_moved:{maxmove:.2f}A")

    dh = adslab.get_distances(ns, list(range(ns)), mic=True)
    zz = adslab.get_atomic_numbers()
    near = int(np.argmin(dh))
    lim = 1.3 * (covalent_radii[zz[ns]] + covalent_radii[zz[near]])
    if dh[near] > lim:
        flags.append(f"H_desorbed:{dh[near]:.2f}A")
    return flags, {"n_formed": nf, "n_broken": nb, "maxmove": round(maxmove, 3)}


dirs = sorted(d for d in glob.glob(os.path.join(RES, "*")) if os.path.isdir(d))
print("reclassifying %d structures (hysteresis %.1f/%.1f, min delta-d %.1f A)\n"
      % (len(dirs), MULT_BOND, MULT_TOL, MIN_DELTA))
print("%-26s %6s %10s %10s %8s" % ("structure", "cand.", "anom_OLD", "anom_NEW", "change"))
print("-" * 66)

tot_old = tot_new = tot_c = 0
zero_new = []
for d in dirs:
    nm = os.path.basename(d)
    csvp = os.path.join(d, "candidates.csv")
    cf = os.path.join(CLEAN, nm + ".traj")
    if not os.path.isfile(csvp) or not os.path.isfile(cf):
        continue
    rows = list(csv.DictReader(open(csvp)))
    if not rows:
        continue
    clean = read(cf)
    old_anom = sum(1 for r in rows if r.get("is_anomalous") == "True")
    new_anom = 0
    for r in rows:
        tp = r["traj_path"]
        if not os.path.isfile(tp):
            r["anomalies_hyst"] = "missing_traj"; r["is_anomalous_hyst"] = "True"
            new_anom += 1; continue
        fl, _ = classify(clean, read(tp))
        r["anomalies_hyst"] = "|".join(fl)
        r["is_anomalous_hyst"] = str(bool(fl))
        if fl:
            new_anom += 1
    cols = list(rows[0].keys())
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    tot_old += old_anom; tot_new += new_anom; tot_c += len(rows)
    if new_anom == len(rows):
        zero_new.append(nm)
    mark = ""
    if old_anom == len(rows) and new_anom < len(rows):
        mark = "  <- was 100 %, fixed"
    print("%-26s %6d %10d %10d %8s%s"
          % (nm, len(rows), old_anom, new_anom, "%+d" % (new_anom - old_anom), mark))

print("-" * 66)
print("TOTAL candidates %d | anomalous OLD %d (%.1f %%) -> NEW %d (%.1f %%)"
      % (tot_c, tot_old, 100 * tot_old / max(tot_c, 1), tot_new, 100 * tot_new / max(tot_c, 1)))
if zero_new:
    print("\n! structures with NO usable candidate even after the fix (%d) - genuinely problematic:"
          % len(zero_new))
    for n in zero_new:
        print("   ", n)
else:
    print("\nOK: every structure has at least one usable candidate")
