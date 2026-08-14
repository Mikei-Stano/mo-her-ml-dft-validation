# ML/DFT validation pipeline for hydrogen adsorption energies

Computational pipeline for validating machine-learning interatomic potentials
against density functional theory when screening hydrogen evolution reaction
(HER) catalysts.

Candidate structures across seven families of Mo-based materials
(MoP, MoS₂, MoSe₂, MoB, Mo₂C, γ-Mo₂N, Ti₃C₂O₂) are screened with AdsorbML/UMA,
and a validation subset is recomputed with full DFT (GPAW, RPBE) so the two
can be compared on equal terms. The validation set is a balanced factorial
design — 7 families × 3 facets × 4 variants — deliberately *not* a top-N
selection from the screening, which would bias the evaluation toward
structures the model itself already favours.

> **This repository contains the code, not the results.** A manuscript
> reporting the findings is in preparation. Result data, figures and the
> analysis layer will be added here, with a DOI, on publication.

## Layout

```
scripts/                   the computational pipeline
  adsorbml/
    1-relax_uma_omat.py    clean-slab relaxation, UMA `omat` task head
    2-run_adsorbml.py      100 H placements per slab, UMA `oc20` task head
    3-extract_rank.py      candidate ranking
  generate_structures.py   bulk crystals and slabs for all seven families
  generate_campaign_set.py the balanced 7 × 3 × 4 factorial design
  gpaw_h_adsorption.py     the DFT driver
  campaign_stage3_rank.py  ranking for the production campaign
hpc/                       SLURM submission, worker scripts, Singularity def
validation/                independent checks — bulk structures against
                           published crystallographic data, AdsorbML anomaly
                           detection with hysteresis, cost model
data/inputs/               the campaign specification (which material, facet
                           and variant to compute) and cell sizes
```

## Method

**DFT** — GPAW, LCAO/dzp basis, RPBE exchange–correlation functional, grid
spacing h = 0.16 Å, Fermi–Dirac smearing σ = 0.1 eV extrapolated to σ → 0,
BFGS relaxation to f_max ≤ 0.05 eV/Å. Symmetry:
`{point_group: False, time_reversal: True}`.

**Reference state** — ΔG_H = E(adslab) − E(clean slab) − ½·E(H₂) + ZPE, with
E(H₂) computed at the same level of theory (RPBE/dzp, relaxed H–H distance)
and ZPE = +0.24 eV.

**ML** — `uma-m-1p1` checkpoint via `fairchem-core`. Model weights are gated
on Hugging Face and are not redistributed here; see
[facebookresearch/fairchem](https://github.com/facebookresearch/fairchem).

Note that the two stages use different UMA task heads by design — `omat` for
clean-surface energetics, `oc20` for adsorption configurations. Mixing them
across a single reference difference is a known pitfall; see
`scripts/adsorbml/` for how the two stages are kept separate.

**Hardware** — developed for the PERUN cluster of the Slovak Academy of
Sciences: AMD EPYC 9845 CPU nodes (320 cores/node) and NVIDIA GH200
Grace-Hopper GPU nodes, InfiniBand NDR.

## Running it

```bash
pip install -r requirements.txt
```

**Structure generation** (local, cheap):

```bash
python3 scripts/generate_structures.py      # bulk crystals and slabs
python3 scripts/generate_campaign_set.py    # the 7 × 3 × 4 factorial set
python3 validation/verify_bulk_fixes.py     # check bulks against experiment
```

**ML screening** (GPU nodes):

```bash
python3 scripts/adsorbml/1-relax_uma_omat.py
python3 scripts/adsorbml/2-run_adsorbml.py
python3 scripts/adsorbml/3-extract_rank.py
```

**DFT validation** (CPU nodes). Cluster login, host and paths come from
environment variables — see the header of `hpc/00_transfer_to_perun.sh`:

```bash
PERUN_USER=<login> bash hpc/00_transfer_to_perun.sh
bash hpc/01_pull_images.sh
sbatch hpc/submit_perun_dft_array.sh
```

## Verification

`validation/verify_bulk_fixes.py` regenerates all seven bulk crystal
structures and checks atom count, composition, minimum interatomic distance
and density against published crystallographic data. Every generator carries
its literature source in the docstring.

`validation/adsorbml_anomaly.py` and `reclassify_anomalies.py` implement an
anomaly detector for adsorption trajectories — bond formation and breaking
with a hysteresis band, slab-atom displacement, and desorption — following the
approach used in the AdsorbML protocol.

## Citation

Please cite via `CITATION.cff`. This section will be updated with the
manuscript DOI on publication.

## License

MIT (`LICENSE`) for code; CC BY 4.0 (`LICENSE-DATA`) for data files.

## Acknowledgement

Computational resources were provided by the Computing Centre of the Slovak
Academy of Sciences (PERUN cluster), project `p2061-26-2`.
