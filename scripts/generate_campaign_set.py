"""
Generátor VYVÁŽENEJ DFT kampane — 7 rodín × 3 fazety × 4 varianty = 84 štruktúr.

PREČO SAMOSTATNÝ SKRIPT (a nie zmena generate_structures.py):
Pôvodný generátor produkoval extrémne nerovnaké pokrytie (Mo2C 40 štruktúr vs
MoP 3), takže dopantová/vakančná os existovala len pre Mo2C/Mo2N/MoB. To je
zároveň vedecká slabina (nedá sa tvrdiť „rodiny × fazety × modifikácie") aj
cenový problém (Mo2C sám by zožral 79 % FAT rozpočtu). Tento skript definuje
kampaň explicitne, s pevným faktoriálnym dizajnom a s per-fazetu optimalizovanými
veľkosťami buniek. Pôvodný generátor zostáva nedotknutý.

DIZAJN (3 × 4 faktoriál, uniformný pre všetkých 7 rodín):
    fazety  : (100), (110), (111)
    varianty: base            — pristinný povrch
              vac<A>          — jedna aniónová vakancia (pre HER najrelevantnejšie
                                aktívne miesto, napr. S-vakancia v MoS2)
              vac2<A>         — dvojitá aniónová vakancia
              dopPt           — substitúcia jedného povrchového kovu Pt
                                (Pt = referenčný HER kov)

VEĽKOSTI BUNIEK: per (rodina, fazeta) z data/optimal_cell_sizes.json — najmenší
počet atómov pri layers>=4 (spodná polovica zamrznutá → >=2 voľné vrstvy) a oboch
in-plane hranách >= 7.0 Å. Prah 7 Å: publikované HER štúdie bežne používajú
3×3 fcc(111) ≈ 7.5 Å; pre malý adsorbát ako H je to obhájiteľné a znižuje cenu
Mo2C 8× (192 → 48/96 atómov na fazetách (111)/(110)).

Použitie:
    CEMEA_CAMPAIGN_OUT=data/inputs/VASP_inputs_campaign \
        python scripts/generate_campaign_set.py
"""
import json
import os
import sys
from pathlib import Path

from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_structures as G  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("CEMEA_CAMPAIGN_OUT",
                          REPO / "data" / "inputs" / "VASP_inputs_campaign"))
SIZES_JSON = REPO / "data" / "optimal_cell_sizes.json"
FACETS = ("(100)", "(110)", "(111)")
DOPANT = os.environ.get("CEMEA_CAMPAIGN_DOPANT", "Pt")

# rodina → (bulk builder, kovová podmriežka, aniónová podmriežka)
FAMILIES = {
    "Mo2C":    (G.create_mo2c_bulk,  "Mo", "C"),
    "Mo2N":    (G.create_mo2n_bulk,  "Mo", "N"),
    "MoB":     (G.create_mob_bulk,   "Mo", "B"),
    "MoS2":    (G.create_mos2_bulk,  "Mo", "S"),
    "MoSe2":   (G.create_mose2_bulk, "Mo", "Se"),
    "MoP":     (G.create_mop_bulk,   "Mo", "P"),
    "Ti3C2O2": (G.create_ti3c2_bulk, "Ti", "O"),
}

# ── optimalizované veľkosti ────────────────────────────────────────────────
SIZES = {}
if SIZES_JSON.is_file():
    for key, v in json.load(open(SIZES_JSON)).items():
        fam, facet = key.split("|")
        SIZES[(fam, facet)] = tuple(v["size"])
else:
    print(f"⚠️  {SIZES_JSON} chýba — použije sa default (2,2,4) pre všetko")


def slab_size(fam, facet):
    return SIZES.get((fam, facet), (2, 2, 4))


def build(fam, facet, variant):
    """Postav jednu štruktúru kampane. Vyhodí výnimku, ak sa nedá (nikdy ticho)."""
    bulk_fn, metal, anion = FAMILIES[fam]
    slab = G.create_slab(bulk_fn(), miller=facet, size=slab_size(fam, facet), vacuum=8)
    if variant == "base":
        return slab
    if variant == f"vac{anion}":
        return G.create_vacancy_slab(slab, anion)
    if variant == f"vac2{anion}":
        return G.create_multi_vacancy_slab(slab, anion, count=2)
    if variant == f"dop{DOPANT}":
        return G.create_substitution_slab(slab, metal, DOPANT)
    raise ValueError(f"neznámy variant {variant}")


def variants_for(fam):
    _, _, anion = FAMILIES[fam]
    return ["base", f"vac{anion}", f"vac2{anion}", f"dop{DOPANT}"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, failed = [], []
    print(f"kampaň → {OUT}\ndizajn: 7 rodín × 3 fazety × 4 varianty = 84 štruktúr\n")

    for fam in FAMILIES:
        print(f"{fam}:")
        for facet in FACETS:
            sz = slab_size(fam, facet)
            for variant in variants_for(fam):
                name = f"{fam}_{facet}" + ("" if variant == "base" else f"_{variant}")
                try:
                    atoms = build(fam, facet, variant)
                except Exception as exc:
                    print(f"    {name:28s} ✗ {type(exc).__name__}: {exc}")
                    failed.append((name, f"{type(exc).__name__}: {exc}"))
                    continue
                d = OUT / name
                d.mkdir(parents=True, exist_ok=True)
                write(str(d / "POSCAR"), atoms, format="vasp")
                manifest.append(dict(
                    name=name, family=fam, facet=facet, variant=variant,
                    size=str(sz), n_atoms=len(atoms),
                    formula=atoms.get_chemical_formula(),
                    n_fixed=sum(len(c.get_indices()) for c in atoms.constraints
                                if c.__class__.__name__ == "FixAtoms"),
                ))
                print(f"    {name:28s} ✓ {len(atoms):4d} at.  {atoms.get_chemical_formula():14s} size={sz}")

    import csv
    mpath = OUT.parent / "campaign_manifest.csv"
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)

    print(f"\n{'='*70}")
    print(f"vygenerovaných: {len(manifest)} / 84   ·   zlyhalo: {len(failed)}")
    if failed:
        print("ZLYHANIA (hlasné, nie tiché — treba doriešiť):")
        for n, why in failed:
            print(f"   {n:28s} {why}")
    tot = sum(m["n_atoms"] for m in manifest)
    print(f"spolu atómov: {tot}, medián {sorted(m['n_atoms'] for m in manifest)[len(manifest)//2]}")

    # odhad ceny (kalibrácia: 96 at. = 2.1 core·h/BFGS krok pri 16 rankoch)
    cost = sum(2.1 * (m["n_atoms"] / 96.0) ** 3 * 60 * 2 * 5 for m in manifest)
    print(f"odhad DFT (60 krokov/strana, billing ×5): ~{cost:.0f} core·h "
          f"= {100*cost/513450:.1f} % zo zostatku 513 450")
    print(f"manifest → {mpath}")


if __name__ == "__main__":
    main()
