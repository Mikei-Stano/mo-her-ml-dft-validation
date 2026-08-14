"""Cena zostávajúcej práce — z NAMERANÝCH single-pointov, nie z extrapolácie starých logov.

KALIBRAČNÉ BODY (val3, h=0.16, mriežka 4x4x1, single-point PÁR clean+adslab,
fixná UMA geometria — teda presne tá istá fyzika ako produkcia, len bez relaxácie):
    MoP_(111)     30 at.   750 s na 16 rankoch  =    3.33 core·h / pár
    Mo2N_(100)    96 at. 15803 s na 32 rankoch  =  140.47 core·h / pár
Z toho exponent skálovania: ln(140.47/3.33) / ln(96/30) = 3.2
To sedí s tým, že v LCAO dominuje diagonalizácia O(N_bázia^3) na k-bod
(GPAW timer: Orbital Layouts 21.6 % + DenseAtomicCorrection 20.5 % + h^-3 časť 55.8 %).

RELAXAČNÝ FAKTOR je najväčšia neistota a je označený ako taký. Skladá sa z:
  - počet iónových krokov: z necenzurovanej podmnožiny 128 BFGS logov je pri
    fmax=0.05 Kaplan-Meier medián 29 krokov (clean), adslab krivka nedosiahne
    50 % → medián > 30. Rozpočet musí stáť na p90-p95, nie na mediáne.
  - cena kroku voči SVIEŽEMU single-pointu: každý ďalší krok štartuje z hustoty
    predošlého, takže SCF konverguje za menej iterácií. Bez merania na h=0.16
    beriem 0.40-0.55.
"""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ── kalibrácia ───────────────────────────────────────────────────────────────
CAL = [(30, 750 * 16 / 3600), (96, 15803 * 32 / 3600)]   # (atómov, core·h/pár)
import math
EXP = math.log(CAL[1][1] / CAL[0][1]) / math.log(CAL[1][0] / CAL[0][0])
A = CAL[0][1] / CAL[0][0] ** EXP


def sp_pair(n):
    """core·h na single-point pár (clean+adslab) pri h=0.16, 4x4x1."""
    return A * n ** EXP


# ── štruktúry kampane ────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(REPO / "data/adsorbml_results/ranked_candidates_campaign.csv")))
CONVERGED = {"MoP", "MoS2", "MoSe2", "Ti3C2O2"}      # 4x4x1 dokázane stačí
METALLIC = {"MoB", "Mo2C", "Mo2N"}                    # potrebujú 6x6x1 (n_IBZ 8→18)
K_PENALTY = 18 / 8                                    # 2.25x viac k-bodovej práce
K_SHARE = 0.62                                        # podiel k-závislej práce (namerané)

# 6x6x1 nezdražuje celý výpočet 2.25x, len jeho k-závislú časť
METAL_FACTOR = (1 - K_SHARE) + K_SHARE * K_PENALTY

STEPS = {"optimisticky": (24, 0.40), "realne": (40, 0.48), "pesimisticky": (60, 0.55)}

print("=" * 78)
print(f"KALIBRÁCIA:  core·h/pár = {A:.4g} · N^{EXP:.2f}     (2 namerané body)")
print(f"             kovové rodiny × {METAL_FACTOR:.2f} za 6x6x1 mriežku")
print("=" * 78)

groups = {"A (konvergované, 4x4x1)": CONVERGED, "B (kovové, 6x6x1)": METALLIC}
sp_tot = {}
for label, fams in groups.items():
    sel = [r for r in rows if r["family"] in fams]
    fac = METAL_FACTOR if fams is METALLIC else 1.0
    tot = sum(sp_pair(int(r["n_atoms_slab"])) * fac for r in sel)
    sp_tot[label] = tot
    ns = sorted(int(r["n_atoms_slab"]) for r in sel)
    print(f"\n{label}:  {len(sel)} štruktúr, {ns[0]}–{ns[-1]} atómov (medián {ns[len(ns)//2]})")
    print(f"  Σ single-point párov: {tot:8.0f} core·h")
    for r in sorted(sel, key=lambda r: -sp_pair(int(r['n_atoms_slab'])))[:3]:
        n = int(r["n_atoms_slab"])
        print(f"    najdrahšie: {r['slab_name']:<24} {n:3d} at.  {sp_pair(n)*fac:8.0f} core·h/pár")

print("\n" + "=" * 78)
print("PLNÁ RELAXÁCIA (obe strany), podľa rozpočtu krokov — ODHAD, nie meranie")
print("=" * 78)
print(f"  {'scenár':<14}{'krokov/strana':>14}{'cena/krok':>11}"
      f"{'kampaň A':>12}{'kampaň B':>12}{'SPOLU':>12}")
for name, (steps, frac) in STEPS.items():
    mult = steps * frac
    a = sp_tot["A (konvergované, 4x4x1)"] * mult
    b = sp_tot["B (kovové, 6x6x1)"] * mult
    print(f"  {name:<14}{steps:>14}{frac:>11.2f}{a:>12.0f}{b:>12.0f}{a+b:>12.0f}")

print("\n" + "=" * 78)
FAT_LEFT = 513597
print(f"ZOSTATOK FAT: {FAT_LEFT} core·h")
for name, (steps, frac) in STEPS.items():
    mult = steps * frac
    t = (sp_tot["A (konvergované, 4x4x1)"] + sp_tot["B (kovové, 6x6x1)"]) * mult
    print(f"  {name:<14} {t:8.0f} core·h = {100*t/FAT_LEFT:5.1f} % zostatku,"
          f"  rezerva {FAT_LEFT/t:.1f}×")
print("=" * 78)
