#!/usr/bin/env python
"""
ULOHA A: kanonicky AdsorbML konektivitny detektor anomalii povrchu
(`has_surface_changed` z ocdata/fairchem, Lan et al. npj Comput Mater 9, 172 (2023)).

Princip
-------
dGH = E(adslab) - E(clean_slab) - 1/2 E(H2) je platne LEN ak relaxovany clean slab
a slabova cast relaxovaneho adslabu reprezentuju TU ISTU povrchovu strukturu.
AdsorbML to testuje konektivitnou maticou:

  1) C_clean  = konektivita relaxovaneho CLEAN slabu
  2) C_adslab = konektivita relaxovaneho ADSLABU, ale iba na PODMNOZINE slabovych
                atomov (adsorbat H vynechany, t.j. tag != 2)
  3) vazba (i,j) existuje ak  d_mic(i,j) < mult * (r_cov(i) + r_cov(j))
     (ASE: natural_cutoffs(atoms, mult) + NeighborList, PBC/mic-aware)
  4) ak sa C zmenila v LUBOVOLNOM smere (vznik ALEBO zanik vazby) -> FLAGGED

Dva varianty kroku 3-4:

  (A) KANONICKY (ocdata DetectTrajAnomaly.has_surface_changed) - cushion je
      TOLERANCIA, pouzita ASYMETRICKY, aby sa neflagovali marginalne pary
      lezice na hrane cutoffu:
        vznik  vazby: C_adslab(mult=1.0) AND NOT C_clean (mult=cushion)
        zanik  vazby: C_clean (mult=1.0) AND NOT C_adslab(mult=cushion)
      t.j. vazba sa pocita za novu len ak je v adslabe kratsia ako 1.0*(ri+rj)
      A ZAROVEN v clean slabe dlhsia ako 1.5*(ri+rj).

  (B) SYMETRICKY (naivne citanie zadania): jeden jediny cutoff mult=cushion
      pre obe matice, akykolvek rozdiel -> flag. Uvedeny pre porovnanie,
      ukazuje sa ako prehnane citlivy na kovovych/kov-bohatych povrchoch.

Vystup: data/outputs/adsorbml_anomaly.json
"""
import os
import csv
import json
import math
from collections import Counter

import numpy as np
from ase.io import read
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.geometry import find_mic

ROOT = os.path.expanduser("~/cemea/mo-h-adsorption-gpaw")
ML_VS_DFT = os.path.join(ROOT, "data/outputs/ml_vs_dft.csv")
RANKED = os.path.join(ROOT, "data/adsorbml_results/ranked_candidates.perun.csv")
MOB = os.path.join(ROOT, "data/adsorbml_results/mob_candidates.perun.csv")
CLEAN_DIR = os.path.join(ROOT, "data/uma_relaxed")
CORR = os.path.join(ROOT, "data/outputs/corr_rmsd_results.json")
OUT = os.path.join(ROOT, "data/outputs/adsorbml_anomaly.json")

CUSHION = 1.5          # AdsorbML cushion / tolerancia (zadanie + ocdata default)
BOND_MULT = 1.0        # ocdata default mult pre "skutocnu" vazbu
CUSHION_SCAN = [1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]
RMSD_THRESH = 0.3      # nas ad-hoc geometricky filter


# ----------------------------------------------------------------- I/O helpers
def load_candidate_map():
    m, src = {}, {}
    for path, tag in ((RANKED, "ranked"), (MOB, "mob")):
        if not os.path.exists(path):
            print("WARN: chyba %s" % path)
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                name = row["slab_name"]
                cf = (row.get("candidate_file") or "").strip()
                if not cf or name in m:
                    continue
                m[name] = cf
                src[name] = tag
    return m, src


def resolve(p):
    if os.path.isabs(p) and os.path.exists(p):
        return p
    base = p
    for marker in ("data/adsorbml_results", "data/uma_relaxed"):
        i = p.find(marker)
        if i >= 0:
            base = p[i:]
            break
    return os.path.join(ROOT, base)


# ------------------------------------------------------- AdsorbML connectivity
def get_connectivity(atoms, cushion=CUSHION):
    """Konektivitna matica podla AdsorbML: d(i,j) < cushion*(r_cov_i + r_cov_j).

    natural_cutoffs(atoms, mult) vracia r_cov[Z]*mult per atom; NeighborList
    spoji i-j ak d <= cutoff_i + cutoff_j = mult*(r_i + r_j). PBC/mic je
    zabezpecene skrz cell+pbc objektu atoms (self_interaction=False).
    """
    cutoffs = natural_cutoffs(atoms, mult=cushion)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True,
                      skin=0.0)
    nl.update(atoms)
    return np.asarray(nl.get_connectivity_matrix(sparse=False)).astype(bool)


def _pairs(mask_upper):
    iu = np.triu_indices(mask_upper.shape[0], k=1)
    return [(int(i), int(j)) for i, j in zip(*iu) if mask_upper[i, j]]


def bond_diff_canonical(Cc_tight, Cc_loose, Ca_tight, Ca_loose):
    """Kanonicky ocdata test s cushionom ako toleranciou.

    formed: vazba je v adslabe pri mult=1.0 a NIE JE v clean slabe ani pri cushione
    broken: vazba je v clean slabe pri mult=1.0 a NIE JE v adslabe ani pri cushione
    """
    formed = _pairs(Ca_tight & ~Cc_loose)
    broken = _pairs(Cc_tight & ~Ca_loose)
    return formed, broken


def bond_diff_symmetric(C_clean, C_ads):
    formed = _pairs(C_ads & ~C_clean)
    broken = _pairs(C_clean & ~C_ads)
    return formed, broken


def describe(formed, broken, symbols, pos_clean, pos_ads, cell, pbc, top_n=12):
    def dist(p, i, j):
        dv = np.array([p[j] - p[i]])
        _, dl = find_mic(dv, cell, pbc=pbc)
        return float(dl[0])

    def pairinfo(pairs):
        return [dict(i=i, j=j, pair="%s-%s" % (symbols[i], symbols[j]),
                     r_sum=float(covalent_r(symbols[i]) + covalent_r(symbols[j])),
                     d_clean=dist(pos_clean, i, j),
                     d_adslab=dist(pos_ads, i, j)) for i, j in pairs]

    tf = Counter("-".join(sorted((symbols[i], symbols[j]))) for i, j in formed)
    tb = Counter("-".join(sorted((symbols[i], symbols[j]))) for i, j in broken)
    return dict(formed_types=dict(tf), broken_types=dict(tb),
                formed_examples=pairinfo(formed[:top_n]),
                broken_examples=pairinfo(broken[:top_n]))


def covalent_r(sym):
    from ase.data import covalent_radii, atomic_numbers
    return covalent_radii[atomic_numbers[sym]]


# ------------------------------------------------------------------ statistics
def rankdata(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), float)
    ranks[order] = np.arange(1, len(v) + 1)
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def main():
    cmap, csrc = load_candidate_map()
    with open(ML_VS_DFT) as fh:
        rows = list(csv.DictReader(fh))
    print("ml_vs_dft.csv riadkov: %d ; candidate map: %d" % (len(rows), len(cmap)))

    rmsd_map = {}
    if os.path.exists(CORR):
        with open(CORR) as fh:
            for d in json.load(fh)["results"]:
                rmsd_map[d["name"]] = dict(rmsd=d["rmsd"], dmax=d["dmax"],
                                           family=d.get("family", ""))

    results, skipped, warns = [], [], []

    for r in rows:
        name = r["slab_name"]
        fam = r["chemistry_group"]
        absd = float(r["abs_delta_eV"])
        delta = float(r["delta_DFT_minus_ML_eV"])

        cf = cmap.get(name)
        if cf is None:
            skipped.append((name, "chyba candidate_file")); continue
        cf = resolve(cf)
        cleanf = os.path.join(CLEAN_DIR, name + ".traj")
        if not os.path.exists(cf):
            skipped.append((name, "chyba adslab: %s" % cf)); continue
        if not os.path.exists(cleanf):
            skipped.append((name, "chyba clean: %s" % cleanf)); continue

        clean = read(cleanf)     # posledny (relaxovany) frame
        ads = read(cf)
        nc, na = len(clean), len(ads)
        sc = clean.get_chemical_symbols()
        sa = ads.get_chemical_symbols()
        note = []

        hidx = [i for i, s in enumerate(sa) if s == "H"]
        if "H" in sc:
            note.append("clean slab obsahuje H!")
        if len(hidx) != 1:
            note.append("H atomov v adslabe: %d" % len(hidx))
        elif hidx[0] != na - 1:
            note.append("H nie je posledny (idx %d/%d)" % (hidx[0], na))
        if na != nc + 1:
            note.append("pocet: clean=%d adslab=%d" % (nc, na))

        # AdsorbML: podmnozina slabovych atomov adslabu (tag != 2, t.j. bez H)
        slab_idx = [i for i in range(na) if i not in set(hidx)]
        nslab = len(slab_idx)
        if nslab != nc:
            note.append("nslab(adslab)=%d != nc=%d" % (nslab, nc))
        m = min(nslab, nc)
        mismatch = sum(1 for k in range(m) if sc[k] != sa[slab_idx[k]])
        if mismatch:
            note.append("symbol mismatch %d/%d" % (mismatch, m))

        ads_slabonly = ads[slab_idx]

        rec = dict(name=name, family=fam, nslab=nslab,
                   absd=absd, delta=delta,
                   rmsd=rmsd_map.get(name, {}).get("rmsd"),
                   dmax=rmsd_map.get(name, {}).get("dmax"),
                   src=csrc.get(name, "?"))

        cell = ads_slabonly.get_cell()
        pbc = ads_slabonly.get_pbc()
        pos_c = clean.get_positions()
        pos_a = ads_slabonly.get_positions()

        # --- tesne matice (mult = BOND_MULT = 1.0) ---
        Cc_t = get_connectivity(clean, BOND_MULT)
        Ca_t = get_connectivity(ads_slabonly, BOND_MULT)
        rec["n_bonds_clean"] = int(Cc_t.sum() // 2)
        rec["n_bonds_adslab"] = int(Ca_t.sum() // 2)

        # --- (A) KANONICKY ocdata test, cushion = tolerancia ---
        Cc_l = get_connectivity(clean, CUSHION)
        Ca_l = get_connectivity(ads_slabonly, CUSHION)
        f, b = bond_diff_canonical(Cc_t, Cc_l, Ca_t, Ca_l)
        rec["bonds_formed"] = len(f)
        rec["bonds_broken"] = len(b)
        rec["flagged"] = bool(f or b)
        rec["detail"] = describe(f, b, sc, pos_c, pos_a, cell, pbc)

        # --- (B) SYMETRICKY variant pri roznych cutoffoch ---
        for cush in (1.0, 1.5):
            fs, bs = bond_diff_symmetric(get_connectivity(clean, cush),
                                         get_connectivity(ads_slabonly, cush))
            k = "sym_c%.1f" % cush
            rec[k + "_formed"] = len(fs)
            rec[k + "_broken"] = len(bs)
            rec[k + "_flagged"] = bool(fs or bs)

        # --- (C) cushion scan pre kanonicky test ---
        scan = {}
        for cush in CUSHION_SCAN:
            fq, bq = bond_diff_canonical(Cc_t, get_connectivity(clean, cush),
                                         Ca_t, get_connectivity(ads_slabonly, cush))
            scan["%.1f" % cush] = dict(formed=len(fq), broken=len(bq),
                                       flagged=bool(fq or bq))
        rec["cushion_scan"] = scan

        rec["note"] = "; ".join(note)
        if note:
            warns.append((name, rec["note"]))
        results.append(rec)

    print("\n=== SPRACOVANE %d, PRESKOCENE %d ===" % (len(results), len(skipped)))
    for n, w in skipped:
        print("  SKIP %s : %s" % (n, w))
    print("=== VAROVANIA (%d) ===" % len(warns))
    for n, w in warns:
        print("  WARN %s : %s" % (n, w))

    # ------------------------------------------------------------ vyhodnotenie
    absd = np.array([d["absd"] for d in results])
    rmsd = np.array([d["rmsd"] if d["rmsd"] is not None else np.nan for d in results])
    names = [d["name"] for d in results]
    fl = np.array([d["flagged"] for d in results])
    rm = rmsd >= RMSD_THRESH

    def mae(mask):
        return float(absd[mask].mean()) if mask.sum() else float("nan")

    def grp(mask):
        return dict(n_flagged=int(mask.sum()), n_unflagged=int((~mask).sum()),
                    mae_flagged=mae(mask), mae_unflagged=mae(~mask),
                    max_absd_unflagged=float(absd[~mask].max()) if (~mask).sum() else None,
                    min_absd_flagged=float(absd[mask].min()) if mask.sum() else None,
                    separation=(mae(mask) / mae(~mask)) if (~mask).sum() and mae(~mask) > 0 else None)

    stats = {
        "adsorbml_canonical_cushion1.5": grp(fl),
        "adsorbml_symmetric_c1.5": grp(np.array([d["sym_c1.5_flagged"] for d in results])),
        "adsorbml_symmetric_c1.0": grp(np.array([d["sym_c1.0_flagged"] for d in results])),
        "rmsd_ge_0.3": grp(rm),
    }
    for c in CUSHION_SCAN:
        k = "%.1f" % c
        stats["canonical_cushion_%s" % k] = grp(
            np.array([d["cushion_scan"][k]["flagged"] for d in results]))

    both = int((fl & rm).sum()); only_a = int((fl & ~rm).sum())
    only_r = int((~fl & rm).sum()); neither = int((~fl & ~rm).sum())

    nbonds = np.array([d["bonds_formed"] + d["bonds_broken"] for d in results], float)
    corr = dict(
        pearson_nbondchange_absd=pearson(nbonds, absd),
        spearman_nbondchange_absd=pearson(rankdata(nbonds), rankdata(absd)),
        pointbiserial_flag_absd=pearson(fl.astype(float), absd),
        spearman_nbondchange_rmsd=pearson(rankdata(nbonds), rankdata(rmsd)),
    )

    # per-family flag rate
    fam_stats = {}
    for d in results:
        s = fam_stats.setdefault(d["family"], dict(n=0, flagged=0, absd=[]))
        s["n"] += 1
        s["flagged"] += int(d["flagged"])
        s["absd"].append(d["absd"])
    for k, s in fam_stats.items():
        s["mae"] = float(np.mean(s.pop("absd")))

    summary = dict(
        cushion=CUSHION, bond_mult=BOND_MULT, rmsd_thresh=RMSD_THRESH,
        n_analyzed=len(results),
        n_flagged_canonical=int(fl.sum()),
        stats=stats,
        confusion_canonical_vs_rmsd=dict(both=both, adsorbml_only=only_a,
                                         rmsd_only=only_r, neither=neither,
                                         agreement=(both + neither) / float(len(results))),
        adsorbml_only_names=[names[i] for i in range(len(names)) if fl[i] and not rm[i]],
        rmsd_only_names=[names[i] for i in range(len(names)) if rm[i] and not fl[i]],
        flagged_names=[names[i] for i in range(len(names)) if fl[i]],
        unflagged_names=[names[i] for i in range(len(names)) if not fl[i]],
        per_family=fam_stats,
        correlations=corr,
    )

    print("\n=== SUMAR ===")
    print(json.dumps(summary, indent=1))

    print("\n=== TABULKA (podla |D|) ===")
    print("| name | family | bonds_formed | bonds_broken | flagged | rmsd [A] | |D| [eV] |")
    print("|---|---|---|---|---|---|---|")
    for d in sorted(results, key=lambda z: z["absd"]):
        print("| %s | %s | %d | %d | %s | %.3f | %.3f |" % (
            d["name"], d["family"], d["bonds_formed"], d["bonds_broken"],
            "YES" if d["flagged"] else "no",
            d["rmsd"] if d["rmsd"] is not None else float("nan"), d["absd"]))

    print("\n=== DETAIL: nezhody (flagged & RMSD<0.3) alebo (unflagged & RMSD>=0.3) ===")
    for d in results:
        disc = (d["flagged"] and d["rmsd"] < RMSD_THRESH) or \
               ((not d["flagged"]) and d["rmsd"] >= RMSD_THRESH)
        if disc:
            print("  %-24s rmsd=%.3f |D|=%.3f formed=%d broken=%d  %s | %s" % (
                d["name"], d["rmsd"], d["absd"], d["bonds_formed"], d["bonds_broken"],
                d["detail"]["formed_types"], d["detail"]["broken_types"]))

    with open(OUT, "w") as fh:
        json.dump(dict(summary=summary, results=results,
                       skipped=skipped, warns=warns), fh, indent=1)
    print("\nJSON -> %s" % OUT)


if __name__ == "__main__":
    main()
