"""
PREKLASIFIKOVANIE ANOMÁLIÍ — oprava falošných pozitív z jedného symetrického prahu.

PROBLÉM (diagnostikovaný): môj prvý detektor používal JEDEN prah (cushion 1.2)
symetricky. Väzba, ktorá leží presne naň, sa pri akejkoľvek relaxácii preklopí →
flag pri každom kandidátovi. OVERENÉ na MoB_(111)_vac2B: Mo–B 2.885 → 2.797 Å pri
prahu 2.856, teda zmena len 0.064–0.088 Å, a citlivosť c=1.2 → 5/5 flagnutých
ale c=1.3 → 2/5 (nekonzistentné = artefakt, nie fyzika).

RIEŠENIE — HYSTERÉZA, ako to robí kanonický ocdata `DetectTrajAnomaly`:
  vznik  = viazané v adslabe pri mult=BOND (1.0) A NEviazané v cleane pri mult=TOL (1.3)
  zánik  = viazané v cleane pri mult=BOND (1.0) A NEviazané v adslabe pri mult=TOL (1.3)
Medzi 1.0 a 1.3 je „mŕtva zóna", v ktorej sa nič neflagne → hraničné väzby neblikajú.
NAVYŠE požadujem, aby sa vzdialenosť reálne zmenila o >= MIN_DELTA (0.2 Å) — väzba
sa nemôže „vytvoriť" pohybom o 0.08 Å.

Beží na CPU nad už vypočítanými candidate_*.traj — NEPOTREBUJE GPU ani nový beh.
Prepíše stĺpce anomálií v candidates.csv a nechá pôvodné pre porovnanie.
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
MULT_BOND, MULT_TOL = 1.0, 1.3     # hysteréza
MIN_DELTA = 0.2                    # Å — minimálna reálna zmena vzdialenosti
MAX_SLAB_MOVE = 1.5                # Å


def bonded(atoms, mult):
    d = atoms.get_all_distances(mic=True)
    r = np.array([covalent_radii[n] for n in atoms.get_atomic_numbers()])
    thr = mult * (r[:, None] + r[None, :])
    np.fill_diagonal(d, np.inf)
    return (d < thr), d


def classify(clean, adslab):
    """Anomálie s hysterézou. Vráti (flags, diag)."""
    ns = len(clean)
    if len(adslab) != ns + 1:
        return ["atom_count_mismatch"], {}
    sub = adslab[:ns]

    b0_hard, d0 = bonded(clean, MULT_BOND)
    b0_soft, _ = bonded(clean, MULT_TOL)
    b1_hard, d1 = bonded(sub, MULT_BOND)
    b1_soft, _ = bonded(sub, MULT_TOL)

    # vznik: pevne viazané v adslabe A ani mäkko neviazané v cleane
    formed = np.triu(b1_hard & ~b0_soft, 1)
    broken = np.triu(b0_hard & ~b1_soft, 1)

    # + požiadavka reálnej zmeny vzdialenosti
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
print("preklasifikujem %d štruktúr (hysteréza %.1f/%.1f, min Δd %.1f Å)\n"
      % (len(dirs), MULT_BOND, MULT_TOL, MIN_DELTA))
print("%-26s %6s %10s %10s %8s" % ("štruktúra", "kand.", "anom_STARÉ", "anom_NOVÉ", "zmena"))
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
        mark = "  ← bolo 100 %, opravené"
    print("%-26s %6d %10d %10d %8s%s"
          % (nm, len(rows), old_anom, new_anom, "%+d" % (new_anom - old_anom), mark))

print("-" * 66)
print("SPOLU kandidátov %d | anomálnych STARÉ %d (%.1f %%) → NOVÉ %d (%.1f %%)"
      % (tot_c, tot_old, 100 * tot_old / max(tot_c, 1), tot_new, 100 * tot_new / max(tot_c, 1)))
if zero_new:
    print("\n⚠ štruktúry BEZ použiteľného kandidáta aj po oprave (%d) — reálne problémové:"
          % len(zero_new))
    for n in zero_new:
        print("   ", n)
else:
    print("\n✓ každá štruktúra má aspoň jedného použiteľného kandidáta")
