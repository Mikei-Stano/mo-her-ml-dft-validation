"""
KAMPAŇ stage 3' — ranking kandidátov pre DFT vstup.

DVE OPRAVY oproti pôvodnému `3-extract_rank.py`:

1) VÝBER KANDIDÁTA. Pôvodný riadok 52 robí
       best_idx = df["gibbs_free_ml_eV"].abs().idxmin()
   t.j. vyberá kandidáta NAJBLIŽŠIEHO K ΔG=0. To je HER-screeningová heuristika, nie
   fyzika — a tlačí ML hodnoty k nule Z KONŠTRUKCIE. OVERENÉ na 14 627 kandidátoch:
       |ΔG| min      → σ(ML ΔG) = 0.067 eV, 39/43 v pásme |ΔG| < 0.05
       min E_adslab  → σ(ML ΔG) = 0.522 eV, len 7/43
       DFT           → σ = 1.634 eV
   Pri σ(ML)=0.067 vs σ(DFT)=1.634 (24×) nie je R² definovateľné — reportované
   R²=0.034 bolo bezcenné z definície. AdsorbML konvencia = NAJNIŽŠIA energia adslabu.

2) VYRADENIE ANOMÁLIÍ. Povinný krok AdsorbML protokolu (Lan et al. 2023), ktorý sa
   v tomto projekte nikdy nevykonal (nativý detektor dával 0 anomálií z 13 803).
   Tu sa vyberá len z kandidátov s `is_anomalous_hyst == False`.

Vstup : data/adsorbml_results_campaign/<name>/candidates.csv
Výstup: data/adsorbml_results/ranked_candidates_campaign.csv
        (stĺpce kompatibilné s gpaw_h_adsorption.py --adsorbml-candidates)
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

# ── VÄZBOVÉ SITO ─────────────────────────────────────────────────────────────
# H musí byť naväzaný na povrch, inak ΔG_H nemeria chemisorpciu. Kandidáti sú
# energeticky degenerovaní (namerané rozdiely 0.33-17.6 meV medzi naväzaným
# a nenaväzaným), takže samotné "min E_adslab" nestačí — pri takom šume vyberie
# ktorýkoľvek z nich. Prah 1.35 × (r_cov(H)+r_cov(X)) je štandardný väzbový
# cushion; voľba nie je citlivá, lebo namerané naväzané kandidáty majú pomer
# 1.00-1.10 a nenaväzané 2.0-2.9.
BOND_CUSHION = 1.35


def _h_bond_ratio(traj_path):
    """d(H–najbližší) / (r_cov(H)+r_cov(X)). None ak sa nedá načítať.
    H je konvenciou posledný atóm adslabu (kontroluje to už stage 2)."""
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

ZPE = 0.24     # ΔG_H = E_ads + 0.24 eV (ZPE + entropia), aplikované RAZ, tu

meta = {r["name"]: r for r in csv.DictReader(open(MAN))} if MAN.is_file() else {}

rows, skipped, stats = [], [], []
for d in sorted(os.listdir(RES)):
    p = RES / d / "candidates.csv"
    if not p.is_file():
        skipped.append((d, "chýba candidates.csv")); continue
    cands = list(csv.DictReader(open(p)))
    if not cands:
        skipped.append((d, "prázdny candidates.csv")); continue

    anom_col = "is_anomalous_hyst" if "is_anomalous_hyst" in cands[0] else "is_anomalous"
    clean_c = [c for c in cands if c.get(anom_col) != "True"]
    if not clean_c:
        skipped.append((d, f"všetkých {len(cands)} kandidátov anomálnych")); continue

    def fnum(c, k):
        try:
            return float(c[k])
        except (KeyError, TypeError, ValueError):
            return None

    # AdsorbML konvencia: najnižšia E_adslab z NEanomálnych
    usable = [c for c in clean_c if fnum(c, "E_adslab_ml_eV") is not None]
    # Väzbové sito: z neanomálnych ponechaj len tie, kde je H reálne naväzaný.
    # Ak by sito vyradilo všetko, radšej sa vráť k neprefiltrovanej množine
    # a označ to v selection_rule — mlčky vypadnutá štruktúra je horšia než
    # štruktúra s poznámkou.
    bonded = [c for c in usable if _is_bonded(c)]
    gate_applied = bool(bonded)
    if gate_applied:
        usable = bonded
    if not usable:
        skipped.append((d, "žiadna platná E_adslab")); continue
    best = min(usable, key=lambda c: fnum(c, "E_adslab_ml_eV"))

    # pre porovnanie: čo by dal pôvodný predpis |ΔG| min (na tých istých dátach)
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


print(f"zapísaných: {len(rows)}  →  {OUT}")
print(f"preskočených: {len(skipped)}")
for n, why in skipped:
    print(f"   {n:28s} {why}")

dgs = [r["gibbs_free_ml_eV"] for r in rows if r["gibbs_free_ml_eV"] != ""]
olds = [r["gibbs_if_old_rule_eV"] for r in rows if r["gibbs_if_old_rule_eV"] != ""]
print("\n=== VPLYV VÝBEROVÉHO PREDPISU (na tých istých dátach) ===")
for lbl, v in (("min E_adslab (správne)", dgs), ("|ΔG| min (pôvodné)", olds)):
    if v:
        n, mean, sd = summ(v)
        near = sum(1 for x in v if abs(x) < 0.1)
        print(f"  {lbl:24s} n={n}  ⟨ΔG⟩={mean:+.3f}  σ={sd:.3f}  |ΔG|<0.1: {near}")
if stats:
    n, mean, sd = summ(stats)
    print(f"  stredný rozdiel medzi predpismi: {mean:.3f} eV, max {max(stats):.3f} eV")

print("\n=== ANOMÁLIE ===")
ta = sum(r["n_anomalous"] for r in rows); tc = sum(r["n_candidates"] for r in rows)
print(f"  vyradených {ta} z {tc} kandidátov ({100*ta/max(tc,1):.1f} %)")
print(f"  štruktúry s <10 použiteľnými kandidátmi: "
      f"{[r['slab_name'] for r in rows if r['n_usable'] < 10]}")

print("\n=== POKRYTIE PO RODINÁCH ===")
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
