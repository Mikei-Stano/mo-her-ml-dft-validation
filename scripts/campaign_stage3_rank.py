"""
CAMPAIGN stage 3 - ranking candidates for the DFT input.

TWO CORRECTIONS relative to the original `3-extract_rank.py`:

1) CANDIDATE SELECTION. The original does
       best_idx = df["gibbs_free_ml_eV"].abs().idxmin()
   i.e. it picks the candidate CLOSEST TO dG = 0. That is an HER screening
   heuristic, not physics, and it pushes the ML values toward zero BY
   CONSTRUCTION. VERIFIED over 14,627 candidates:
       |dG| min      -> sigma(ML dG) = 0.067 eV, 39/43 inside |dG| < 0.05
       min E_adslab  -> sigma(ML dG) = 0.522 eV, only 7/43
       DFT           -> sigma = 1.634 eV
   With sigma(ML) = 0.067 against sigma(DFT) = 1.634 (a factor of 24) an R^2 is
   not meaningfully defined, so the reported R^2 = 0.034 was worthless by
   construction. The AdsorbML convention is the LOWEST adslab energy.

2) ANOMALY REJECTION. A mandatory step of the AdsorbML protocol (Lan et al.
   2023) that was never actually performed in this project - the native
   detector reported 0 anomalies out of 13,803. Here the selection is made only
   among candidates with `is_anomalous_hyst == False`.

Input : data/adsorbml_results_campaign/<name>/candidates.csv
Output: data/adsorbml_results/ranked_candidates_campaign.csv
        (columns compatible with gpaw_h_adsorption.py --adsorbml-candidates)
        data/outputs/campaign_stage3_report.json
"""
import csv
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "data/adsorbml_results_campaign"
CLEAN = REPO / "data/uma_relaxed_campaign"
MAN = REPO / "data/inputs/campaign_manifest.csv"
OUT = REPO / "data/adsorbml_results/ranked_candidates_campaign.csv"
REPORT = REPO / "data/outputs/campaign_stage3_report.json"

# -- BOND FILTER -------------------------------------------------------------
# H must be bound to the surface, otherwise dG_H does not measure chemisorption.
# The candidates are energetically degenerate - measured differences of
# 0.33-17.6 meV between bound and unbound - so "min E_adslab" alone is not
# enough: at that noise level it picks either one. The threshold
# 1.35 x (r_cov(H) + r_cov(X)) is a standard bonding cushion; the choice is not
# sensitive, since measured bound candidates give ratios of 1.00-1.10 and
# unbound ones 2.0-2.9.
BOND_CUSHION = 1.35


def _h_bond_ratio(traj_path):
    """d(H-nearest) / (r_cov(H) + r_cov(X)). None if it cannot be read.
    By convention H is the last atom of the adslab, as checked in stage 2."""
    try:
        import numpy as np
        from ase.data import atomic_numbers, covalent_radii
        from ase.io import read
        a = read(str(traj_path))
        if a.get_chemical_symbols()[-1] != "H":
            return None
        d = np.linalg.norm(a.positions[:-1] - a.positions[-1], axis=1)
        j = int(np.argmin(d))
        exp = (covalent_radii[atomic_numbers["H"]]
               + covalent_radii[atomic_numbers[a[j].symbol]])
        return float(d[j] / exp)
    except Exception:
        return None


def _is_bonded(cand):
    r = _h_bond_ratio(cand.get("traj_path", ""))
    return r is not None and r < BOND_CUSHION

ZPE = 0.24     # dG_H = E_ads + 0.24 eV (ZPE + entropy), applied ONCE, here

meta = {r["name"]: r for r in csv.DictReader(open(MAN))} if MAN.is_file() else {}

rows, skipped, stats = [], [], []
for d in sorted(os.listdir(RES)):
    p = RES / d / "candidates.csv"
    if not p.is_file():
        skipped.append((d, "candidates.csv missing")); continue
    cands = list(csv.DictReader(open(p)))
    if not cands:
        skipped.append((d, "empty candidates.csv")); continue

    anom_col = "is_anomalous_hyst" if "is_anomalous_hyst" in cands[0] else "is_anomalous"
    clean_c = [c for c in cands if c.get(anom_col) != "True"]
    if not clean_c:
        skipped.append((d, f"all {len(cands)} candidates anomalous")); continue

    def fnum(c, k):
        try:
            return float(c[k])
        except (KeyError, TypeError, ValueError):
            return None

    # AdsorbML convention: the lowest E_adslab among the NON-anomalous ones
    usable = [c for c in clean_c if fnum(c, "E_adslab_ml_eV") is not None]
    # Bond filter: among the non-anomalous ones keep only those where H is
    # genuinely bound. If the filter would remove everything, fall back to the
    # unfiltered set and record that in selection_rule - a silently dropped
    # structure is worse than a structure carrying a caveat.
    bonded = [c for c in usable if _is_bonded(c)]
    gate_applied = bool(bonded)
    if gate_applied:
        usable = bonded
    if not usable:
        skipped.append((d, "no valid E_adslab")); continue
    best = min(usable, key=lambda c: fnum(c, "E_adslab_ml_eV"))

    # for comparison: what the original |dG| min rule would give on the same data
    with_ads = [c for c in usable if fnum(c, "E_ads_ml_eV") is not None]
    old_best = min(with_ads, key=lambda c: abs(fnum(c, "E_ads_ml_eV") + ZPE)) if with_ads else None

    e_ads = fnum(best, "E_ads_ml_eV")
    dg = (e_ads + ZPE) if e_ads is not None else None
    dg_old = (fnum(old_best, "E_ads_ml_eV") + ZPE) if old_best else None
    m = meta.get(d, {})

    rows.append({
        "slab_name": d,
        "slab_file": str(CLEAN / f"{d}.traj"),
        "candidate_file": best["traj_path"],
        "gibbs_free_ml_eV": round(dg, 6) if dg is not None else "",
        "E_ads_ml_eV": round(e_ads, 6) if e_ads is not None else "",
        "E_adslab_ml_eV": round(fnum(best, "E_adslab_ml_eV"), 6),
        "best_rank": best.get("candidate_rank", ""),
        "family": m.get("family", ""), "facet": m.get("facet", ""),
        "variant": m.get("variant", ""), "n_atoms_slab": m.get("n_atoms", ""),
        "n_candidates": len(cands),
        "n_anomalous": sum(1 for c in cands if c.get(anom_col) == "True"),
        "n_usable": len(usable),
        "gibbs_if_old_rule_eV": round(dg_old, 6) if dg_old is not None else "",
        "selection_rule": ("min_E_adslab_nonanomalous_bonded" if gate_applied
                           else "min_E_adslab_nonanomalous_NO_BONDED_CANDIDATE"),
    })
    if dg is not None and dg_old is not None:
        stats.append(abs(dg - dg_old))

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)


def summ(vals):
    n = len(vals); mean = sum(vals) / n
    sd = (sum((x - mean) ** 2 for x in vals) / n) ** 0.5
    return n, mean, sd


print(f"written: {len(rows)}  ->  {OUT}")
print(f"skipped: {len(skipped)}")
for n, why in skipped:
    print(f"   {n:28s} {why}")

dgs = [r["gibbs_free_ml_eV"] for r in rows if r["gibbs_free_ml_eV"] != ""]
olds = [r["gibbs_if_old_rule_eV"] for r in rows if r["gibbs_if_old_rule_eV"] != ""]
print("\n=== EFFECT OF THE SELECTION RULE (on the same data) ===")
for lbl, v in (("min E_adslab (correct)", dgs), ("|dG| min (original)", olds)):
    if v:
        n, mean, sd = summ(v)
        near = sum(1 for x in v if abs(x) < 0.1)
        print(f"  {lbl:24s} n={n}  ⟨ΔG⟩={mean:+.3f}  σ={sd:.3f}  |ΔG|<0.1: {near}")
if stats:
    n, mean, sd = summ(stats)
    print(f"  mean difference between rules: {mean:.3f} eV, max {max(stats):.3f} eV")

print("\n=== ANOMALIES ===")
ta = sum(r["n_anomalous"] for r in rows); tc = sum(r["n_candidates"] for r in rows)
print(f"  rejected {ta} of {tc} candidates ({100*ta/max(tc,1):.1f} %)")
print(f"  structures with fewer than 10 usable candidates: "
      f"{[r['slab_name'] for r in rows if r['n_usable'] < 10]}")

print("\n=== COVERAGE BY FAMILY ===")
from collections import Counter
for fam, c in sorted(Counter(r["family"] for r in rows).items()):
    print(f"  {fam or '?':12s} {c:2d}/12")

REPORT.write_text(json.dumps({
    "n_ranked": len(rows), "n_skipped": len(skipped),
    "skipped": [{"name": n, "reason": w} for n, w in skipped],
    "selection_rule": ("min_E_adslab among non-anomalous (hysteresis detector) "
                       f"AND H bonded: d < {BOND_CUSHION} x (r_cov(H)+r_cov(X)))"),
    "zpe_correction_eV": ZPE,
    "sigma_new": summ(dgs)[2] if dgs else None,
    "sigma_old_rule": summ(olds)[2] if olds else None,
    "anomalous_total": ta, "candidates_total": tc,
}, indent=2))
print(f"\nreport → {REPORT}")
