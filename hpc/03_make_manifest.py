"""
FÁZA 1c — vygeneruje AdsorbML manifest pre všetky štruktúry z ranked_structures.csv
(spusti na PERUNe, stačí login uzol — čistý stdlib Python).

- slab_file ukazuje na data/uma_relaxed/<meno>.traj TOHTO repa (aktuálne cesty),
- Millerove indexy sa parsujú z názvu štruktúry, napr. "MoB_(110)_sheet" -> (1, 1, 0),
- štruktúry bez slab .traj preskočí s varovaním (očakávaný 1 duplikát *_Node1),
- zálohuje data/adsorbml_manifest.csv a nahradí ho plnou verziou
  (číta ho stage 3 extract_rank; stage 2 hotové štruktúry sám preskakuje).

Použitie:
    python3 hpc/03_make_manifest.py
    (voliteľne --ranked-csv /ina/cesta/ranked_structures.csv)
"""

import argparse
import csv
import os
import re
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MILLER_RE = re.compile(r"_\((\d)(\d)(\d)\)")


def millers_from_name(name: str) -> str:
    m = MILLER_RE.search(name)
    if not m:
        return "(0, 0, 1)"
    return f"({m.group(1)}, {m.group(2)}, {m.group(3)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked-csv",
                     default=os.path.join(os.path.dirname(REPO_ROOT), "ranked_structures.csv"))
    args = ap.parse_args()

    uma_dir = os.path.join(REPO_ROOT, "data", "uma_relaxed")
    manifest_path = os.path.join(REPO_ROOT, "data", "adsorbml_manifest.csv")
    ads_dir = os.path.join(REPO_ROOT, "data", "adsorbml_results")

    ranked = [r["slab_name"] for r in csv.DictReader(open(args.ranked_csv))]

    rows, skipped, done = [], [], 0
    for name in ranked:
        slab_file = os.path.join(uma_dir, name + ".traj")
        if not os.path.isfile(slab_file):
            skipped.append(name)
            continue
        if os.path.isfile(os.path.join(ads_dir, name, "candidates.csv")):
            done += 1
        rows.append({"slab_name": name,
                     "slab_file": slab_file,
                     "millers": millers_from_name(name)})

    if os.path.isfile(manifest_path):
        shutil.copy(manifest_path, manifest_path + ".bak")
        print(f"Zálohoval som pôvodný manifest -> {manifest_path}.bak")

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slab_name", "slab_file", "millers"])
        w.writeheader()
        w.writerows(rows)

    print(f"Manifest: {len(rows)} štruktúr -> {manifest_path}")
    print(f"  z toho už hotových (stage 2 ich preskočí): {done}")
    print(f"  na dopočítanie: {len(rows) - done}")
    if skipped:
        print(f"  PRESKOČENÉ (chýba slab .traj): {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
