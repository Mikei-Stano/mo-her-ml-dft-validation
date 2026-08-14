"""AdsorbML step 3: rank slabs by the ML-predicted adsorption free energy, pick
the best H* candidate per slab, extract its H position and write a manifest for
the DFT driver.

ATTRIBUTION
    Written by Michal Skalican (github.com/skalican19) as part of
    github.com/misohu/mo-h-adsorption-gpaw, maintained by Michal Hucko
    (github.com/misohu). Included here unmodified.

    Note: this script selects the candidate closest to dG = 0. That is a
    screening heuristic rather than the AdsorbML convention of lowest adslab
    energy; see scripts/campaign_stage3_rank.py in this repository for the
    alternative used in the validation campaign, and the reasoning behind it.
"""
from pathlib import Path

import pandas as pd
from ase.io import read

REPO_ROOT          = Path(__file__).resolve().parents[2]
OUT_DIR            = REPO_ROOT / "data" / "adsorbml_results"
MANIFEST_CSV       = REPO_ROOT / "data" / "adsorbml_manifest.csv"
RANKED_CSV         = OUT_DIR / "ranked_candidates.csv"
ENTROPY_CORRECTION = 0.24  # eV, standard ZPE+entropy correction for H*


def main():
    manifest = pd.read_csv(MANIFEST_CSV).set_index("slab_name")

    rows = []
    skipped = []

    for candidates_csv in sorted(OUT_DIR.glob("*/candidates.csv")):
        slab_name = candidates_csv.parent.name

        try:
            df = pd.read_csv(candidates_csv)
        except Exception as exc:
            skipped.append((slab_name, str(exc)))
            continue

        if df.empty or "E_ads_ml_eV" not in df.columns:
            skipped.append((slab_name, "empty or missing E_ads_ml_eV"))
            continue

        df = df.dropna(subset=["E_ads_ml_eV"])
        if df.empty:
            skipped.append((slab_name, "all E_ads_ml_eV are NaN"))
            continue

        df["gibbs_free_ml_eV"] = df["E_ads_ml_eV"] + ENTROPY_CORRECTION
        best_idx = df["gibbs_free_ml_eV"].abs().idxmin()
        best = df.loc[best_idx]

        # H atom is always the last atom in the adslab trajectory
        try:
            adslab = read(str(best["traj_path"]))
            h_pos  = adslab[-1].position
        except Exception as exc:
            print(f"  WARN {slab_name}: could not read H position: {exc}")
            h_pos = [float("nan")] * 3

        slab_file = (
            manifest.loc[slab_name, "slab_file"]
            if slab_name in manifest.index else ""
        )

        rows.append({
            "slab_name":        slab_name,
            "best_rank":        int(best["candidate_rank"]),
            "E_ads_ml_eV":      float(best["E_ads_ml_eV"]),
            "gibbs_free_ml_eV": float(best["gibbs_free_ml_eV"]),
            "h_x":              float(h_pos[0]),
            "h_y":              float(h_pos[1]),
            "h_z":              float(h_pos[2]),
            "slab_file":        slab_file,
            "candidate_file":   str(best["traj_path"]),
        })

    if not rows:
        print("No results found. Run 2-run_adsorbml.py first.")
        return

    ranked = pd.DataFrame(rows).sort_values("gibbs_free_ml_eV", key=abs).reset_index(drop=True)
    ranked.to_csv(RANKED_CSV, index=False)
    print(f"\nRanked candidates written: {RANKED_CSV}  ({len(ranked)} slabs)")

    if skipped:
        print(f"\nSkipped {len(skipped)} slabs:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    print(f"\nTop 20 by |ΔG*H| (ML):")
    cols = ["slab_name", "gibbs_free_ml_eV", "E_ads_ml_eV", "best_rank"]
    print(ranked.head(20)[cols].to_string(index=False))


if __name__ == "__main__":
    main()