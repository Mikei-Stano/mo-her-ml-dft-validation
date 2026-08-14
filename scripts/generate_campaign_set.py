"""
Generator for a BALANCED DFT campaign: 7 families x 3 facets x 4 variants = 84 structures.

WHY A SEPARATE SCRIPT (rather than changing generate_structures.py):
the original generator produced extremely uneven coverage - 40 structures for
Mo2C against 3 for MoP - so the dopant and vacancy axes existed only for
Mo2C/Mo2N/MoB. That is both a scientific weakness (one cannot claim a
"families x facets x modifications" design) and a cost problem, since Mo2C
alone would consume 79 % of the allocation. This script defines the campaign
explicitly, with a fixed factorial design and per-facet optimised cell sizes.
The original generator is left untouched.

DESIGN (a 3 x 4 factorial, uniform across all seven families):
    fazety  : (100), (110), (111)
    variants: base            - pristine surface
              vac<A>          - single anion vacancy (the most HER-relevant
                                active site, e.g. an S vacancy in MoS2)
              vac2<A>         - double anion vacancy
              dopPt           - substitution of one surface metal atom by Pt
                                (Pt as the reference HER metal)

CELL SIZES: taken per (family, facet) from data/optimal_cell_sizes.json - the
smallest atom count with layers >= 4 (the lower half is frozen, leaving >= 2
free layers) and both in-plane edges >= 7.0 A. On the 7 A threshold: published
HER studies commonly use a 3x3 fcc(111) cell of about 7.5 A; for an adsorbate
as small as H that is defensible, and it cuts the Mo2C cost eightfold
(192 -> 48/96 atoms on the (111)/(110) facets).

Usage:
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

# family -> (bulk builder, metal sublattice, anion sublattice)
FAMILIES = {
    "Mo2C":    (G.create_mo2c_bulk,  "Mo", "C"),
    "Mo2N":    (G.create_mo2n_bulk,  "Mo", "N"),
    "MoB":     (G.create_mob_bulk,   "Mo", "B"),
    "MoS2":    (G.create_mos2_bulk,  "Mo", "S"),
    "MoSe2":   (G.create_mose2_bulk, "Mo", "Se"),
    "MoP":     (G.create_mop_bulk,   "Mo", "P"),
    "Ti3C2O2": (G.create_ti3c2_bulk, "Ti", "O"),
}

# -- optimised cell sizes ---------------------------------------------------
SIZES = {}
if SIZES_JSON.is_file():
    for key, v in json.load(open(SIZES_JSON)).items():
        fam, facet = key.split("|")
        SIZES[(fam, facet)] = tuple(v["size"])
else:
    print(f"!  {SIZES_JSON} missing - falling back to the default (2,2,4) everywhere")


def slab_size(fam, facet):
    return SIZES.get((fam, facet), (2, 2, 4))


def build(fam, facet, variant):
    """Build one campaign structure. Raises if it cannot - never fails silently."""
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
    raise ValueError(f"unknown variant {variant}")


def variants_for(fam):
    _, _, anion = FAMILIES[fam]
    return ["base", f"vac{anion}", f"vac2{anion}", f"dop{DOPANT}"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, failed = [], []
    print(f"campaign -> {OUT}\ndesign: 7 families x 3 facets x 4 variants = 84 structures\n")

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
    print(f"generated: {len(manifest)} / 84   ·   failed: {len(failed)}")
    if failed:
        print("FAILURES (loud, not silent - these need resolving):")
        for n, why in failed:
            print(f"   {n:28s} {why}")
    tot = sum(m["n_atoms"] for m in manifest)
    print(f"total atoms: {tot}, median {sorted(m['n_atoms'] for m in manifest)[len(manifest)//2]}")

    # cost estimate (calibration: 96 atoms = 2.1 core-h per BFGS step on 16 ranks)
    cost = sum(2.1 * (m["n_atoms"] / 96.0) ** 3 * 60 * 2 * 5 for m in manifest)
    print(f"odhad DFT (60 krokov/strana, billing ×5): ~{cost:.0f} core·h "
          f"= {100*cost/513450:.1f} % zo zostatku 513 450")
    print(f"manifest → {mpath}")


if __name__ == "__main__":
    main()
