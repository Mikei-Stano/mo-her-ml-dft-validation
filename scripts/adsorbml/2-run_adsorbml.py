"""
scripts/adsorbml/2-run_adsorbml.py

AdsorbML step 2: Screen H* adsorption candidates on UMA-M relaxed slabs using
AdsorbML (100 placements per slab, ML-relaxed with UMA-M OC20 head).

Reads:  data/adsorbml_manifest.csv
Writes: data/adsorbml_results/<name>/candidates.csv
        data/adsorbml_results/<name>/candidate_*.traj
        data/adsorbml_results/<name>/adsorbml.log
        data/adsorbml_results/batch_summary.csv

Usage:
  python scripts/adsorbml/2-run_adsorbml.py
"""
import ast
import argparse
import glob
import logging
import os
import subprocess
import traceback
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import ase.io
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
from fairchem.data.oc.core import Slab
from fairchem.core.components.calculate.recipes.adsorbml import run_adsorbml
from fairchem.core import FAIRChemCalculator

REPO_ROOT        = Path(__file__).resolve().parents[2]
MANIFEST_CSV     = REPO_ROOT / "data" / "adsorbml_manifest.csv"
OUT_DIR          = REPO_ROOT / "data" / "adsorbml_results"
ADSORBATE_SMILES = "*H"
MIN_FREE_VRAM_GB = 8.0
WORKERS_PER_GPU  = 1

_CANDIDATES_COLS = [
    "candidate_rank", "E_adslab_ml_eV", "E_slab_ml_eV",
    "E_gas_ref_ml_eV", "E_ads_ml_eV", "anomalies", "traj_path",
]

_MANIFEST_REQUIRED_COLS = {"slab_name", "slab_file", "millers"}

master_log = logging.getLogger("adsorbml.master")
_LOG_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
_COMPOUND_LOG_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


def _setup_logging(log_path=None) -> None:
    master_log.setLevel(logging.DEBUG)
    if master_log.handlers:
        return
    ch = logging.StreamHandler()
    ch.setFormatter(_LOG_FMT)
    master_log.addHandler(ch)
    if log_path:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(_LOG_FMT)
        master_log.addHandler(fh)


def _parse_millers(millers_str) -> tuple:
    if isinstance(millers_str, tuple):
        return millers_str
    try:
        v = ast.literal_eval(str(millers_str))
        return v if isinstance(v, tuple) else (0, 0, 1)
    except Exception:
        return (0, 0, 1)


def _is_done(slab_name: str) -> bool:
    return (OUT_DIR / slab_name / "candidates.csv").exists()


def _close_log(log: logging.Logger) -> None:
    for h in log.handlers[:]:
        h.close()
        log.removeHandler(h)


def _resolve_slab_file_path(raw_path: str) -> tuple[str, bool]:
    """Resolve slab_file paths across machines into local workspace paths."""
    path = Path(str(raw_path)).expanduser()
    if path.exists():
        return str(path), False

    # Common case for failed manifests: absolute path from another machine.
    parts = path.parts
    if "uma_relaxed" in parts:
        uma_idx = parts.index("uma_relaxed")
        rel_after_uma = Path(*parts[uma_idx + 1:])
        mapped = REPO_ROOT / "data" / "uma_relaxed" / rel_after_uma
        if mapped.exists():
            return str(mapped), True

    mapped_by_name = REPO_ROOT / "data" / "uma_relaxed" / path.name
    if mapped_by_name.exists():
        return str(mapped_by_name), True

    return str(path), False


def _load_manifest_rows(manifest_csv: Path) -> list[dict]:
    df = pd.read_csv(manifest_csv)
    missing_cols = _MANIFEST_REQUIRED_COLS.difference(df.columns)
    if missing_cols:
        raise ValueError(
            f"Manifest missing required columns: {sorted(missing_cols)}"
        )

    rows = df.to_dict("records")
    valid_rows = []
    remapped = 0
    missing_paths = []

    for row in rows:
        raw_path = row.get("slab_file", "")
        resolved_path, was_remapped = _resolve_slab_file_path(raw_path)
        if was_remapped:
            remapped += 1

        if not Path(resolved_path).exists():
            missing_paths.append((row.get("slab_name", "<unknown>"), raw_path, resolved_path))
            continue

        normalized = dict(row)
        normalized["slab_file"] = resolved_path
        valid_rows.append(normalized)

    master_log.info(f"Manifest: {manifest_csv}")
    master_log.info(
        "Manifest rows: %d  |  Valid slab paths: %d  |  Remapped: %d  |  Missing: %d",
        len(rows), len(valid_rows), remapped, len(missing_paths)
    )

    preview = 10
    for slab_name, raw_path, resolved_path in missing_paths[:preview]:
        master_log.warning(
            "Missing slab file for %s: raw='%s' resolved='%s'",
            slab_name, raw_path, resolved_path,
        )
    if len(missing_paths) > preview:
        master_log.warning("... and %d more missing slab paths", len(missing_paths) - preview)

    return valid_rows


def process_row(row: dict, calc) -> None:
    slab_file = row["slab_file"]
    slab_name = row["slab_name"]
    millers   = _parse_millers(row["millers"])

    run_dir  = OUT_DIR / slab_name
    done_csv = run_dir / "candidates.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    comp_log = logging.getLogger(f"adsorbml.{slab_name}")
    comp_log.setLevel(logging.DEBUG)
    comp_log.propagate = False
    if not comp_log.handlers:
        fh = logging.FileHandler(str(run_dir / "adsorbml.log"), mode="a", encoding="utf-8")
        fh.setFormatter(_LOG_FMT)
        comp_log.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(_COMPOUND_LOG_FMT)
        comp_log.addHandler(ch)

    if done_csv.exists():
        comp_log.info(f"SKIP (already done): {slab_name}")
        _close_log(comp_log)
        return

    comp_log.info("=" * 50)
    comp_log.info(f"Running: {slab_name}")

    try:
        atoms = ase.io.read(slab_file)
        # tag=2 is reserved for adsorbate atoms; clamp any slab atoms mistakenly tagged 2 → 0
        tags = atoms.get_tags()
        tags[tags == 2] = 0
        atoms.set_tags(tags)
        if not atoms.constraints:
            atoms.set_constraint(FixAtoms(mask=[t == 0 for t in atoms.get_tags()]))
        slab  = Slab(bulk=None, slab_atoms=atoms, millers=millers,
                     shift=None, top=None, oriented_bulk=None)
    except Exception as exc:
        comp_log.error(f"Could not load slab: {exc}")
        _close_log(comp_log)
        return

    try:
        outputs = run_adsorbml(
            slab=slab,
            adsorbate=ADSORBATE_SMILES,
            calculator=calc,
            optimizer_cls=LBFGS,
            fmax=0.02,
            steps=100,
            num_placements=100,
            reference_ml_energies=True,
        )
    except Exception as exc:
        comp_log.error(f"run_adsorbml failed: {exc}\n{traceback.format_exc()}")
        _close_log(comp_log)
        return

    candidates = outputs["adslabs"]
    if not candidates:
        comp_log.warning(f"No valid placements for {slab_name}")
        pd.DataFrame(columns=_CANDIDATES_COLS).to_csv(done_csv, index=False)
        _close_log(comp_log)
        return

    rows = []
    for i, cand in enumerate(candidates):
        traj_path = str(run_dir / f"candidate_{i}.traj")
        ase.io.write(traj_path, cand["atoms"])
        res = cand["results"]
        ref = res.get("referenced_adsorption_energy", {})
        rows.append({
            "candidate_rank":  i,
            "E_adslab_ml_eV":  res.get("energy", float("nan")),
            "E_slab_ml_eV":    ref.get("slab_energy", float("nan")),
            "E_gas_ref_ml_eV": ref.get("gas_reactant_energy", float("nan")),
            "E_ads_ml_eV":     ref.get("adsorption_energy", float("nan")),
            "anomalies":       "|".join(res.get("adslab_anomalies", [])),
            "traj_path":       traj_path,
        })

    pd.DataFrame(rows).to_csv(done_csv, index=False)
    best_e = min(r["E_ads_ml_eV"] for r in rows)
    comp_log.info(f"Best E_ads (ML) = {best_e:.4f} eV  ({len(candidates)} candidates)")
    _close_log(comp_log)


def _detect_gpus() -> list:
    if not torch.cuda.is_available():
        return []
    eligible = []
    master_log.info("GPU inventory:")
    for i in range(torch.cuda.device_count()):
        props    = torch.cuda.get_device_properties(i)
        free_gb  = torch.cuda.mem_get_info(i)[0] / 1e9
        total_gb = torch.cuda.mem_get_info(i)[1] / 1e9
        ok = free_gb >= MIN_FREE_VRAM_GB
        status = "OK" if ok else f"LOW VRAM – skipped"
        master_log.info(f"  GPU {i}: {props.name}  {free_gb:.1f}/{total_gb:.1f} GB  [{status}]")
        if ok:
            eligible.append((free_gb, i))
    eligible.sort(reverse=True)
    return [i for _, i in eligible]


def _worker(gpu_id, worker_idx: int, task_queue) -> None:
    _setup_logging()
    device = "cuda" if gpu_id is not None else "cpu"
    if gpu_id is not None:
        fraction = 1.0 / WORKERS_PER_GPU
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    master_log.info(f"Loading UMA-M OC20 on {device} (worker {worker_idx})...")
    calc = FAIRChemCalculator.from_model_checkpoint("uma-m-1p1", task_name="oc20", device=device)
    while True:
        row = task_queue.get()
        if row is None:
            break
        process_row(row, calc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AdsorbML candidate screening on a selected manifest."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(MANIFEST_CSV),
        help="Path to manifest CSV (default: data/adsorbml_manifest.csv)",
    )
    args = parser.parse_args()

    manifest_csv = Path(args.manifest).expanduser()
    if not manifest_csv.is_absolute():
        manifest_csv = (REPO_ROOT / manifest_csv).resolve()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = str(OUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    _setup_logging(log_path)
    master_log.info(f"Log: {log_path}")

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    master_log.info("Detecting GPUs...")
    gpu_ids  = _detect_gpus()
    n_workers = len(gpu_ids) * WORKERS_PER_GPU if gpu_ids else 1
    if gpu_ids:
        master_log.info(f"Launching {n_workers} worker(s) across GPU(s): {gpu_ids}")
    else:
        master_log.warning("No eligible GPU — running on CPU.")

    all_rows = _load_manifest_rows(manifest_csv)
    pending  = [r for r in all_rows if not _is_done(r["slab_name"])]
    master_log.info(
        f"Total: {len(all_rows)}  |  Done: {len(all_rows) - len(pending)}  |  Pending: {len(pending)}"
    )

    if not gpu_ids:
        gpu_ids = [None]

    if pending:
        ctx   = mp.get_context("spawn")
        queue = ctx.Queue()
        for row in pending:
            queue.put(row)
        for _ in range(n_workers):
            queue.put(None)

        # CUDA_VISIBLE_DEVICES must be set in the parent before p.start() so
        # the child inherits it before fairchem's import initialises the CUDA
        # runtime (setting it inside the worker function is too late).
        saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        procs = []
        for i in range(n_workers):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            if gpu_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            p = ctx.Process(target=_worker, args=(gpu_id, i, queue))
            p.start()
            procs.append(p)
        if saved_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        for p in procs:
            p.join()

    # Consolidate all per-slab candidates.csv into one batch summary
    all_csvs = sorted(glob.glob(str(OUT_DIR / "*" / "candidates.csv")))
    frames = []
    for csv_path in all_csvs:
        try:
            part = pd.read_csv(csv_path)
            if len(part) > 0:
                part.insert(0, "slab_name", Path(csv_path).parent.name)
                frames.append(part)
        except Exception as exc:
            master_log.warning(f"Skipping {csv_path}: {exc}")

    if frames:
        summary      = pd.concat(frames, ignore_index=True)
        summary_path = OUT_DIR / "batch_summary.csv"
        summary.to_csv(summary_path, index=False)
        master_log.info(f"Saved consolidated results → {summary_path}")

        best = summary.loc[summary.groupby("slab_name")["E_ads_ml_eV"].idxmin(),
                           ["slab_name", "E_ads_ml_eV"]]
        master_log.info("Best ML adsorption energies per slab:\n" + best.to_string(index=False))
    else:
        master_log.info("No results to summarise yet.")
