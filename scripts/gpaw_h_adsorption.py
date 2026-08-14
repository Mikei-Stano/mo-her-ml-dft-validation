"""
GPAW-based H Adsorption Energy Calculator
Calculate ΔGH for Mo compounds using quantum physics simulation

Requirements:
- GPAW (pip install gpaw) ✓ Already installed
- ASE (comes with GPAW)

Features:
- 20-step BFGS relaxation for improved accuracy
- Auto-detecting parallelism (scales to available CPU/RAM)
- Incremental CSV writing (crash-safe)
- Checkpoint/resume (skips already-completed entries)
- Graceful shutdown on SIGINT/SIGTERM
"""

import os
import csv
import json
import signal
import fcntl
import argparse
import fnmatch
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from ase import Atoms
from ase.io import read
from gpaw import GPAW
from gpaw.mpi import world   # MPI world (size=1, rank=0 v sériovom behu)
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from datetime import datetime

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUTS = REPO_ROOT / "data" / "inputs" / "VASP_inputs"
DATA_OUTPUTS = REPO_ROOT / "data" / "outputs"
GPAW_OUTPUTS = DATA_OUTPUTS / "gpaw_calculations"
H2_REFERENCE_FILE = DATA_OUTPUTS / "h2_reference_energy.json"
OUTPUT_CSV = DATA_OUTPUTS / "gpaw_h_adsorption_results_v2.csv"
# ── Prázdne env premenné = "nezadané" ────────────────────────────────────────
# hpc/submit_perun_dft_array.sh exportuje každú voliteľnú premennú aj keď nie je
# zadaná — ako PRÁZDNY string (`GPAW_MODE="${GPAW_MODE:-}"`). os.environ.get(k, d)
# vracia default len keď kľúč CHÝBA, takže prázdno prepíše default: mode sa stalo
# '' namiesto 'lcao', GPAW(mode='') vyhodilo výnimku s prázdnym str() a v logu
# bolo vidno len `✗ Error: ''`. Následok: joby 74524/74571/74619, 97 taskov FAILED.
# Rieši sa RAZ na vstupe, nie pri každom čítaní — inak sa to zopakuje pri každej
# novej premennej pridanej do --export.
for _empty_key in [_k for _k, _v in list(os.environ.items())
                   if not str(_v).strip()
                   and _k.startswith(('GPAW_', 'RELAX_', 'ADSORBML_', 'HUBBARD_'))]:
    del os.environ[_empty_key]

OUTPUT_JSON = DATA_OUTPUTS / "gpaw_h_adsorption_results_v2.json"
ADSORBML_OUTPUT_CSV = Path(os.environ.get(
    "ADSORBML_OUTPUT_CSV", str(DATA_OUTPUTS / "gpaw_adsorbml_results.csv")))

def _env_num(name, default, cast=float):
    """Numerická env premenná, ktorá prežije PRÁZDNY string.

    hpc/submit_perun_dft_array.sh robí `GPAW_X="${GPAW_X:-}"`, čiže premennú vždy
    NASTAVÍ — prípadne na ''. os.environ.get(k, dflt) vracia dflt len keď kľúč
    chýba, takže float('') zhodí každý rank a MPI_Abort zabije job (74524: 47/47
    FAILED za 22 s). Prázdno preto berieme ako "nezadané".

    Nečíselná hodnota sa NEtoleruje — preklep GPAW_H=0,16 musí byť hlasný,
    inak by sa potichu počítalo s iným h, než čo je v provenancii.
    """
    raw = os.environ.get(name, '')
    if raw is None or not str(raw).strip():
        return cast(default)
    try:
        return cast(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}={raw!r} nie je platné číslo: {exc}") from exc


# Configuration
GPAW_CONFIG = {
    'mode': 'lcao',          # Linear Combination of Atomic Orbitals (faster)
    'basis': 'dzp',          # Double-zeta + polarization (good accuracy)
    'xc': 'PBE',             # Exchange-correlation functional
    'kpts': (4, 4, 1),       # K-point mesh for slab
    'txt': 'gpaw.txt',       # Output log file
    'convergence': {
        'energy': 1e-5,      # Energy convergence (eV)
        'density': 1e-4,     # Electron density convergence
        'eigenstates': 1e-5, # Eigenstate convergence
    },
}

# ── Funkcionál a bázia (env-prepínače pre konzistenciu s UMA) ─────
# UMA/OC20 referencia = VASP plane-wave RPBE. Pre like-for-like porovnanie:
#   GPAW_XC=RPBE      → funkcionál (default PBE); RPBE opravuje PBE prevažovanie
#                       väzby adsorbátu (Hammer-Hansen-Nørskov) = UMA úroveň
#   GPAW_MODE=pw      → plnovlnná bázia; robustná pre VEĽKÉ bunky (napr.
#                       512-atómové MoB_(100), kde LCAO/LFC segfaultne) a
#                       najbližšie k VASP-PW referencii. Default 'lcao'
#                       (rýchly; pre slaby s vákuom výrazne lacnejší než PW).
#   GPAW_PW_ECUT=400  → plane-wave cutoff [eV] v PW móde
GPAW_CONFIG['xc'] = os.environ.get('GPAW_XC', GPAW_CONFIG['xc'])
GPAW_CONFIG['mode'] = os.environ.get('GPAW_MODE', GPAW_CONFIG['mode']).strip().lower()
GPAW_PW_ECUT = _env_num('GPAW_PW_ECUT', 400.0)
#   GPAW_H=0.16       → mriežkový rozostup [Å]. Doteraz sa NEnastavoval, takže
#                       bežal implicitný GPAW default ~0.20, ktorý si GPAW volí
#                       sám a zaokrúhľuje gpts na násobky 4 → naprieč kampaňou
#                       reálne 0.197–0.212 Å (rozptyl 7.2 %), do Methods sa nedá
#                       napísať číslo. NAMERANÉ (val3, MoP_(111)): ΔG_H konverguje
#                       ako ~h² k -0.086 eV; h=0.20 je od limity o 0.029 eV,
#                       h=0.18 o 0.017, h=0.16 o 0.007. Produkcia: 0.16.
GPAW_H = _env_num('GPAW_H', 0.20)
# Symetria: vysoko-symetrické bunky (napr. MoP_(100) 128 at., P4mm) padajú na
# GPAW bugu `SymmetryAnalysisBug` v group_check() PRED SCF. `symmetry='off'`
# obíde buggy redukciu — dáva IDENTICKÉ energie (symetria je len urýchlenie),
# cena zanedbateľná pri malej k-mriežke. GPAW_SYMMETRY=off zapne.
GPAW_SYMMETRY = os.environ.get('GPAW_SYMMETRY', '').strip().lower()
# k-body override (FAT-ladenie). Pre VEĽKÉ superbunky (napr. 128-at. MoB_(100))
# je hustá 4×4×1 mriežka drahá aj zbytočná — väčšia bunka = menšia Brillouinova
# zóna, takže redšia k-mriežka stačí. ΔG_H je ROZDIEL energií (E_slab+H − E_slab),
# kde systematická k-chyba do veľkej miery vypadne. GPAW_KPTS="2,2,1".
_kpts_env = os.environ.get('GPAW_KPTS', '').strip()
if _kpts_env:
    GPAW_CONFIG['kpts'] = tuple(int(x) for x in _kpts_env.replace('x', ',').split(','))
# Smearing (Fermi-Diracova šírka). Väčšie σ = hladšie obsadenie pri Fermiho
# hladine → rýchlejšia a stabilnejšia SCF konvergencia kovových povrchov
# (default GPAW 0.1 eV je pre ťažko-konvergujúce kovové MoB primalé). GPAW_SIGMA=0.15.
GPAW_SIGMA = os.environ.get('GPAW_SIGMA', '').strip()

# ── Výkon & presnosť (env-prepínače) ─────────────────────────────
# VŠETKY defaulty zachovávajú súčasné čisté-PBE správanie a NEznižujú
# presnosť. Zrýchlenie (MPI, ScaLAPACK) je EXAKTNÉ — rovnaké k-body,
# cutoff aj báza, len sa práca rozdelí. Voľby, ktoré MENIA fyziku
# (Stichove odporúčania: DFT+U, spin), sú OFF a treba ich vedome
# zapnúť pre SAMOSTATNÚ kampaň (inak pokazia porovnanie s UMA/PBE).
def _env_flag(name, default=False):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")

# (a) ScaLAPACK — paralelná diagonalizácia (exaktné). Účinné len ak je
#     GPAW postavený so ScaLAPACK; inak GPAW voľbu bezpečne ignoruje.
USE_SCALAPACK = _env_flag("GPAW_SCALAPACK", False)

# (b) DFT+U (Hubbard) — Stich: čisté PBE je pre 3d/prechodné kovy
#     nespoľahlivé (lokalizačná chyba d-stavov). MENÍ výsledky -> default
#     prázdne. Env formát: HUBBARD_U="Mo:3.0,Ni:4.0"
def _parse_hubbard_u(spec):
    out = {}
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if tok:
            el, u = tok.split(":")
            out[el.strip()] = ":d,{}".format(float(u))
    return out
HUBBARD_U = _parse_hubbard_u(os.environ.get("HUBBARD_U", ""))

# (c) Spinová polarizácia — Stich: 3d kovy (Ni, Fe, Co, Mn) sú magnetické.
#     MENÍ výsledky a ~2x predražuje -> default OFF (GPAW_SPINPOL=1).
SPINPOL = _env_flag("GPAW_SPINPOL", False)
MAGNETIC_INIT = {"Fe": 2.2, "Co": 1.6, "Ni": 0.6, "Mn": 2.5, "Cr": 1.0}

RELAXATION_CONFIG = {
    'fmax': _env_num('RELAX_FMAX', 0.05),                  # Force conv (eV/Å)
    # Strop krokov: predošlá kampaň mala 30 a ten strop bol BINDING v 27 zo 44
    # výpočtov (finálny fmax 0.068–2.411 eV/Å, energia ešte klesala až 1.28 eV/3
    # kroky) → výsledky nepoužiteľné, ~86k core·h premrhaných. Preto 200 + relax
    # je REŠTARTOVATEĽNÝ (viď _maybe_relax), takže scancel nezničí prácu.
    'steps': _env_num('RELAX_STEPS', 200, int),
}

USE_RELAXATION = True

# AdsorbML mód: default single-point (rýchly). Pre reaktívne kovové povrchy je
# single-point na UMA(RPBE) geometrii nespoľahlivý (ML geom ≠ DFT minimum →
# ΔG mimo o desiatky eV). ADSORBML_RELAX=1 zapne DFT relaxáciu clean aj adslab
# (spodok zamrznutý) → konzistentné DFT minimá = správna adsorpčná energia.
# Odporúčané spolu s GPAW_XC=RPBE, RELAX_FMAX=0.05, RELAX_STEPS=30.
ADSORBML_RELAX = _env_flag("ADSORBML_RELAX", False)

# Parallelism: cores allocated per GPAW calculation
CORES_PER_CALC = 11
# RAM per calculation (GB) for auto-detection
RAM_PER_CALC_GB = 4
# Per-structure hard timeout (hours). 0 disables timeout.
MAX_HOURS_PER_STRUCTURE = 0.0

# CSV header for incremental writes
CSV_COLUMNS = [
    'formula', 'surface_facet', 'adsorbate',
    'E_clean_slab_eV', 'E_slab_with_h_eV', 'E_h2_eV',
    'ΔGH_eV', 'descriptor_eV', 'source',
    'adsorption_site', 'h2_source', 'relaxed',
    'status', 'timestamp',
]

# CSV header for AdsorbML-mode results (separate file, includes ML comparison column)
# PROVENANCE (pridané 2026-07-28): bez týchto stĺpcov sa nedalo odlíšiť konvergovaný
# výpočet od utnutého ani zistiť, akými nastaveniami bol riadok spočítaný — čo umožnilo,
# aby sa do jedného CSV appendovali behy s rôznym XC/k-mriežkou/σ bez rozlíšenia.
ADSORBML_PROVENANCE_COLUMNS = [
    'relax_converged_clean', 'fmax_final_clean', 'relax_steps_clean',
    'relax_converged_adslab', 'fmax_final_adslab', 'relax_steps_adslab',
    'xc', 'mode', 'basis', 'kpts', 'sigma', 'symmetry',
    'relax_fmax_target', 'relax_steps_max', 'spinpol',
    'n_atoms_clean', 'n_atoms_adslab', 'calc_dir', 'config_sha8',
]
ADSORBML_CSV_COLUMNS = CSV_COLUMNS + ['gibbs_free_ml_eV'] + ADSORBML_PROVENANCE_COLUMNS


# Posledný traceback zo spracovania štruktúry — vypisuje sa spolu s err_text.
_LAST_EXC = {}


def _config_fingerprint():
    """Krátky hash celej výpočtovej konfigurácie — ide do každého CSV riadku aj do
    názvu výpočtového adresára, takže kampane sa nemôžu navzájom prepísať."""
    import hashlib
    parts = [
        str(GPAW_CONFIG.get('xc')), str(GPAW_CONFIG.get('mode')),
        str(GPAW_CONFIG.get('basis')), str(GPAW_CONFIG.get('kpts')),
        str(GPAW_SIGMA), str(GPAW_SYMMETRY), str(GPAW_PW_ECUT), str(GPAW_H),
        str(SPINPOL), str(sorted(HUBBARD_U.items())),
        str(ADSORBML_RELAX), str(RELAXATION_CONFIG['fmax']),
        str(RELAXATION_CONFIG['steps']),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]

# ── Graceful shutdown ────────────────────────────────────────────
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n⚠️  Shutdown requested — finishing current calculations, then exiting...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Auto-detect parallelism ─────────────────────────────────────
def _detect_max_workers(override_workers=None):
    """Determine how many parallel GPAW calculations to run."""
    if override_workers is not None:
        print(f"Using manual worker override: {override_workers}")
        return max(1, int(override_workers))

    cpu_count = os.cpu_count() or 4
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break
            else:
                ram_gb = 16
    except OSError:
        ram_gb = 16

    by_cpu = max(1, cpu_count // CORES_PER_CALC)
    by_ram = max(1, int(ram_gb // RAM_PER_CALC_GB))
    workers = min(by_cpu, by_ram)
    print(f"Auto-detected: {cpu_count} CPUs, {ram_gb:.0f} GB RAM → {workers} parallel workers")
    return workers


def _set_thread_env(threads_per_calc):
    """Set thread environment for BLAS/OpenMP libraries."""
    t = max(1, int(threads_per_calc))
    os.environ['OMP_NUM_THREADS'] = str(t)
    os.environ['OPENBLAS_NUM_THREADS'] = str(t)
    os.environ['MKL_NUM_THREADS'] = str(t)
    os.environ['NUMEXPR_NUM_THREADS'] = str(t)
    print(f"Thread env: OMP_NUM_THREADS={t}, OPENBLAS_NUM_THREADS={t}, MKL_NUM_THREADS={t}")


# ── Incremental CSV ──────────────────────────────────────────────
def _ensure_csv_header(csv_path, columns=None):
    """Create CSV with header if it doesn't exist."""
    if world.rank != 0:          # pod MPI zapisuje iba rank 0
        return
    csv_path = Path(csv_path)
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns if columns is not None else CSV_COLUMNS)


def _append_result_csv(result_row, csv_path):
    """Append a single result row to CSV with file locking."""
    if world.rank != 0:          # pod MPI zapisuje iba rank 0 (inak N duplicít)
        return
    csv_path = Path(csv_path)
    with open(csv_path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            writer.writerow(result_row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _load_completed_keys(csv_path):
    """Load set of (formula, surface_facet) already completed in CSV."""
    csv_path = Path(csv_path)
    completed = set()
    if not csv_path.exists():
        return completed
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if row.get('status') == 'completed':
                completed.add((str(row['formula']), str(row['surface_facet'])))
    except Exception:
        pass
    return completed


def discover_structures(base_dir, include_patterns=None):
    """Discover all POSCAR files under base_dir and return labels.

    Args:
        base_dir: directory containing structure subdirectories
        include_patterns: optional list of glob patterns (e.g. ["Ni_Mo2C_*", "Mo2N_*"]).
            If provided, only directories matching at least one pattern are included.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    items = []
    for entry in sorted(base_path.iterdir()):
        if not entry.is_dir():
            continue
        poscar = entry / "POSCAR"
        if not poscar.exists():
            continue
        # Apply include filter
        if include_patterns:
            if not any(fnmatch.fnmatch(entry.name, pat) for pat in include_patterns):
                continue
        parts = entry.name.split("_", 1)
        formula = parts[0]
        surface = parts[1] if len(parts) > 1 else "unknown"
        items.append((formula, surface, entry))
    return items


def filter_structures_by_name(structures, selected_names):
    """Keep only structure directories whose names were explicitly requested."""
    if not selected_names:
        return structures

    requested = [name.strip() for name in selected_names if name and name.strip()]
    if not requested:
        return structures

    requested_set = set(requested)
    filtered = [item for item in structures if item[2].name in requested_set]
    missing = [name for name in requested if name not in {item[2].name for item in filtered}]
    if missing:
        raise ValueError(f"Requested structures not found: {', '.join(missing)}")
    return filtered


def setup_gpaw_calculator(label='gpaw'):
    """
    Create GPAW calculator with optimized settings
    
    Args:
        label: Label for calculation files
        
    Returns:
        GPAW calculator object
    """
    kwargs = dict(
        xc=GPAW_CONFIG['xc'],
        kpts=GPAW_CONFIG['kpts'],
        txt=label + '.txt',
        convergence=GPAW_CONFIG['convergence'],
    )
    # Bázia: LCAO (rýchle, default) alebo plnovlnná PW (robustná pre veľké bunky).
    if GPAW_CONFIG['mode'] == 'pw':
        from gpaw import PW
        kwargs['mode'] = PW(GPAW_PW_ECUT)      # PW nepoužíva LCAO 'basis'
    else:
        kwargs['mode'] = GPAW_CONFIG['mode']   # 'lcao'
        kwargs['basis'] = GPAW_CONFIG['basis']
        kwargs['h'] = GPAW_H                   # inak GPAW volí ~0.20 sám (7.2 % rozptyl)
    # Exaktné zrýchlenie (nemení výsledok): paralelná diagonalizácia.
    if USE_SCALAPACK:
        kwargs['parallel'] = {'sl_auto': True}
    # Stichove voľby (menia fyziku, default OFF):
    if HUBBARD_U:
        kwargs['setups'] = dict(HUBBARD_U)
    if SPINPOL:
        kwargs['spinpol'] = True
    # SYMETRIA — pozor, `symmetry='off'` je v GPAW ekvivalent
    #   Symmetry(point_group=False, time_reversal=False)
    # čo znamená IBZ = celá BZ (overené v logoch: "Number of BZ points: 16 /
    # Number of IBZ points: 16", teda ŽIADNA redukcia).
    #
    # Workaround na SymmetryAnalysisBug (MoP_(100), P4mm) potreboval vypnúť LEN
    # point_group: bug je v group_check() (gpaw/new/symmetry.py), ktorý iteruje
    # výhradne cez `rotation_scc`; pri point_group=False je rotation_scc=[identita],
    # takže sa nemôže vyhodiť. Vypnutie `time_reversal` bola kolaterálna škoda.
    #
    # time_reversal=True je EXAKTNÉ pre nemagnetický systém bez spin-orbitálnej
    # väzby: E_n(k) = E_n(-k) (Kramers) → polovica MP mriežky je redundantná.
    # Overené: dáva n_IBZ = 8 (zo 16) pre VŠETKÝCH 84 štruktúr kampane → 2x menej
    # diagonalizácií pri chybe 0.000 eV. Naviac umožní MPI_RANKS = 8 namiesto 16,
    # čím sa alokácia zmenší z 80 na 40 jadier pri nezmenenom wall-time.
    if GPAW_SYMMETRY in ('off', 'nopg'):
        kwargs['symmetry'] = {'point_group': False, 'time_reversal': True}
    elif GPAW_SYMMETRY == 'hard-off':
        # únikový ventil, ak by point_group=False na nejakej bunke nestačilo
        kwargs['symmetry'] = 'off'
    # Smearing pre kovové povrchy (FAT-ladenie SCF konvergencie):
    if GPAW_SIGMA:
        from gpaw import FermiDirac
        kwargs['occupations'] = FermiDirac(float(GPAW_SIGMA))
    return GPAW(**kwargs)


def _apply_initial_magmoms(atoms):
    """Nastaví štartovacie magnetické momenty pre magnetické 3d kovy
    (len ak je zapnutá spinová polarizácia — Stichovo odporúčanie)."""
    if not SPINPOL:
        return atoms
    moments = [MAGNETIC_INIT.get(sym, 0.0) for sym in atoms.get_chemical_symbols()]
    if any(m != 0.0 for m in moments):
        atoms.set_initial_magnetic_moments(moments)
    return atoms


def _prepare_slab(slab_file):
    """Load slab and enforce minimum vacuum."""
    slab = read(slab_file)
    if slab.cell[2, 2] < 10:
        slab.cell[2, 2] = 15
        slab.center(axis=2, vacuum=0)
    return slab


def _maybe_relax(atoms, label):
    """Local relaxation. REŠTARTOVATEĽNÁ a so ZÁZNAMOM konvergencie.

    Dve opravy proti predošlej kampani:
      (1) `restart=` + `trajectory=` → pri scancel/timeout sa Hessián aj geometria
          zachovajú a ďalší beh pokračuje, nezačína od nuly (predtým sa 27 zo 44
          výpočtov utnulo na strope a práca sa zahodila).
      (2) výsledok konvergencie sa uloží do `atoms.info`, aby ho volajúci mohol
          zapísať do CSV. Bez toho sa nedá odlíšiť konvergovaný výpočet od
          utnutého — a presne to spôsobilo rozstrel E_clean 4.56 eV.
    Vracia `atoms` (rovnaká signatúra ako predtým, aby volajúci nemuseli meniť).
    """
    atoms.info.setdefault('relax_requested', bool(USE_RELAXATION))
    if not USE_RELAXATION:
        atoms.info.update(relax_ran=False, relax_converged=None,
                          relax_steps=0, fmax_final=None)
        return atoms

    # Ensure FixAtoms constraints survived .copy()
    if not atoms.constraints:
        positions = atoms.get_positions()
        z = positions[:, 2]
        z_mid = (np.min(z) + np.max(z)) / 2
        fixed = [i for i in range(len(atoms)) if atoms[i].z < z_mid]
        if fixed:
            atoms.set_constraint(FixAtoms(indices=fixed))

    n_fixed = sum(len(c.get_indices()) for c in atoms.constraints
                  if isinstance(c, FixAtoms))

    optimizer = BFGS(atoms, logfile=f"{label}_relax.log",
                     restart=f"{label}_relax.pckl",      # Hessián prežije restart
                     trajectory=f"{label}_relax.traj")   # geometria prežije restart
    converged = optimizer.run(fmax=RELAXATION_CONFIG['fmax'],
                              steps=RELAXATION_CONFIG['steps'])

    try:
        fmax_final = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())
    except Exception:
        fmax_final = None

    atoms.info.update(
        relax_ran=True,
        relax_converged=bool(converged),
        relax_steps=int(getattr(optimizer, 'nsteps', -1)),
        relax_steps_max=RELAXATION_CONFIG['steps'],
        relax_fmax_target=RELAXATION_CONFIG['fmax'],
        fmax_final=fmax_final,
        n_fixed=n_fixed,
        hit_step_cap=bool(getattr(optimizer, 'nsteps', 0) >= RELAXATION_CONFIG['steps']),
    )
    if not converged:
        print(f"     ⚠️  {os.path.basename(label)}: relax NEKONVERGOVAL "
              f"(kroky {atoms.info['relax_steps']}/{RELAXATION_CONFIG['steps']}, "
              f"fmax {fmax_final}) — riadok bude označený relax_converged=False")
    return atoms


def _candidate_h_positions(slab, h_distance=1.5, max_sites=6):
    """Generate adsorption candidates from top-layer atomic positions."""
    positions = slab.get_positions()
    top_z = np.max(positions[:, 2])
    z_tol = 0.35

    top_indices = [i for i, pos in enumerate(positions) if (top_z - pos[2]) <= z_tol]
    if not top_indices:
        center_xy = np.mean(positions[:, :2], axis=0)
        return [("center_fallback", [center_xy[0], center_xy[1], top_z + h_distance])]

    top_xy = positions[top_indices, :2]
    center_xy = np.mean(top_xy, axis=0)
    distances = [np.linalg.norm(xy - center_xy) for xy in top_xy]
    ranked = [xy for _, xy in sorted(zip(distances, top_xy), key=lambda x: x[0])]

    candidates = []
    for i, xy in enumerate(ranked[:max_sites]):
        candidates.append((f"top_{i+1}", [float(xy[0]), float(xy[1]), float(top_z + h_distance)]))

    # Add one bridge candidate between two most central top atoms when available.
    if len(ranked) >= 2:
        bridge_xy = 0.5 * (ranked[0] + ranked[1])
        candidates.append(("bridge_center", [float(bridge_xy[0]), float(bridge_xy[1]), float(top_z + h_distance)]))

    # Add a centroid candidate for broad coverage.
    candidates.append(("top_centroid", [float(center_xy[0]), float(center_xy[1]), float(top_z + h_distance)]))

    unique = []
    for label, pos in candidates:
        duplicate = any(np.allclose(pos[:2], other_pos[:2], atol=1e-3) for _, other_pos in unique)
        if not duplicate:
            unique.append((label, pos))
    return unique


def calculate_clean_slab_energy(slab, output_dir):
    """
    Calculate energy of clean surface (no adsorbate)
    
    Args:
        slab: ASE Atoms object (already prepared)
        output_dir: Directory to save results
        
    Returns:
        float: Total energy in eV, or None if failed
    """
    
    try:
        print(f"  └─ Calculating clean slab energy...")
        
        slab = slab.copy()
        
        # Setup GPAW calculator
        calc = setup_gpaw_calculator(label=f'{output_dir}/clean_slab')
        slab.calc = calc
        slab = _maybe_relax(slab, f'{output_dir}/clean_slab')
        
        # Get energy
        energy = slab.get_potential_energy()
        print(f"     ✓ Clean slab: E = {energy:.6f} eV")
        
        return energy
    
    except Exception as e:
        print(f"     ✗ Error: {e}")
        return None


def calculate_slab_with_h_energy(slab, output_dir, h_distance=1.5):
    """
    Calculate energy of surface with H adsorbate
    
    Args:
        slab: ASE Atoms object (already prepared)
        output_dir: Directory to save results
        h_distance: Distance of H above surface (Å)
        
    Returns:
        tuple: (best_energy, best_site_label, best_position) or (None, None, None)
    """
    
    try:
        print(f"  └─ Calculating slab+H energy...")
        
        candidates = _candidate_h_positions(slab, h_distance=h_distance)

        best_energy = None
        best_site = None
        best_position = None

        for site_label, h_pos in candidates:
            slab_with_h = slab.copy()
            slab_with_h += Atoms('H', positions=[h_pos])

            calc = setup_gpaw_calculator(label=f'{output_dir}/slab_with_h_{site_label}')
            slab_with_h.calc = calc
            slab_with_h = _maybe_relax(slab_with_h, f'{output_dir}/slab_with_h_{site_label}')

            energy = slab_with_h.get_potential_energy()
            print(f"     · site={site_label:<12} E = {energy:.6f} eV")

            if best_energy is None or energy < best_energy:
                best_energy = energy
                best_site = site_label
                best_position = h_pos

        print(f"     ✓ Best slab+H: site={best_site}, E = {best_energy:.6f} eV")
        return best_energy, best_site, best_position
    
    except Exception as e:
        print(f"     ✗ Error: {e}")
        return None, None, None


# Merania z relaxácie H₂ — plní ich calculate_h2_molecule_energy(),
# číta get_h2_reference_energy() pri zápise provenancie do JSON.
_H2_PROVENANCE = {}


def calculate_h2_molecule_energy(output_dir):
    """
    Calculate energy of isolated H2 molecule
    
    Note: This is expensive, so we use a large vacuum and simpler settings
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        float: Energy of H2 molecule (eV), or None if failed
    """
    
    try:
        print(f"  └─ Calculating H₂ molecule energy...")
        
        # Create H2 molecule in a large box
        h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.75]])
        
        # Add vacuum
        h2.center(vacuum=10)
        
        # Setup GPAW with relaxed k-points (fewer k-points for molecule)
        calc = GPAW(
            mode=GPAW_CONFIG['mode'],
            basis=GPAW_CONFIG['basis'],
            h=GPAW_H,        # KRITICKÉ: ΔG_H = E_adslab - E_clean - ½E(H₂), a H₂ je
                             # v INEJ bunke. Bez rovnakého h je ΔG_H nekonzistentné.
            xc=GPAW_CONFIG['xc'],
            kpts=(1, 1, 1),  # Only 1 k-point for isolated molecule
            txt=f'{output_dir}/h2_molecule.txt',
            convergence=GPAW_CONFIG['convergence'],
        )
        
        h2.calc = calc
        # H₂ MUSÍ byť zrelaxovaná — d = 0.750 Å je ASE default, nie RPBE/dzp minimum.
        # Namerané (job 74292): minimum je d = 0.7757 Å, E = -6.433239 eV proti
        # -6.421482 eV pri 0.750 Å. Tých 11.8 meV je +5.9 meV posun v každom ΔG_H.
        h2_opt = BFGS(h2, logfile=f'{output_dir}/h2_relax.log',
                      trajectory=f'{output_dir}/h2_relax.traj')
        h2_converged = bool(h2_opt.run(fmax=0.01, steps=50))
        energy = h2.get_potential_energy()
        d_hh = float(np.linalg.norm(h2.positions[1] - h2.positions[0]))
        print(f"     ✓ H₂ relaxovaná: d(H–H) = {d_hh:.4f} Å, {h2_opt.nsteps} krokov, "
              f"{'konvergovala' if h2_converged else 'NEKONVERGOVALA (fmax 0.01)'}")
        _H2_PROVENANCE.update(d_HH_A=d_hh, bfgs_steps=int(h2_opt.nsteps),
                              relax_converged=h2_converged)
        print(f"     ✓ H₂ molecule: E = {energy:.6f} eV")
        
        return energy
    
    except Exception as e:
        import traceback
        print(f"     ✗ H₂ zlyhalo: {type(e).__name__}: {e!r}")
        traceback.print_exc()
        return None


def _h2_config_fingerprint():
    """Hash tých parametrov, ktoré MENIA E(H₂). ΔG_H = E_adslab − E_clean − ½E(H₂),
    pričom H₂ sa počíta v INEJ bunke než slaby — takže mriežka `h`, bázia, mód
    a funkcionál musia byť tie isté, inak je ΔG_H nekonzistentné."""
    import hashlib
    parts = [str(GPAW_CONFIG.get('mode')), str(GPAW_CONFIG.get('basis')),
             str(GPAW_CONFIG.get('xc')), f"{GPAW_H:.4f}"]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def get_h2_reference_energy():
    """Get H2 reference energy from cache or compute once for this campaign.

    Cache sa NIKDY nemaže: v array jobe by 47 taskov mazalo a prepočítavalo ten
    istý súbor naraz (race + 47× tá istá práca). Pri nesúlade fingerprintu sa
    beh zastaví s instrukciou, aby sa referencia naseedovala raz — mlčky použiť
    inú referenciu je horšie než spadnúť.
    """
    want = _h2_config_fingerprint()
    if H2_REFERENCE_FILE.exists():
        try:
            with open(H2_REFERENCE_FILE, 'r') as f:
                payload = json.load(f)
            energy = float(payload['E_h2_eV'])
            got = payload.get('h2_config_sha8')
            if got is not None and got != want:
                raise RuntimeError(
                    f"H₂ referencia je pre INÚ konfiguráciu (cache {got}, teraz {want}: "
                    f"mode={GPAW_CONFIG['mode']} basis={GPAW_CONFIG['basis']} "
                    f"xc={GPAW_CONFIG['xc']} h={GPAW_H}). ΔG_H by bolo nekonzistentné. "
                    f"Naseeduj referenciu raz (seedh2.sbatch s rovnakými GPAW_* "
                    f"premennými) a spusti kampaň znova. Súbor: {H2_REFERENCE_FILE}")
            if got is None:
                print(f"⚠️  H₂ cache bez fingerprintu (staršia) — použije sa, "
                      f"ale provenancia je neúplná: {H2_REFERENCE_FILE}")
            print(f"✓ Loaded cached H2 reference: {energy:.6f} eV  [cfg {got or 'n/a'}]")
            return energy, "cache"
        except RuntimeError:
            raise
        except Exception as exc:
            print(f"⚠️  Ignoring invalid H2 cache ({exc}), recomputing...")

    h2_dir = GPAW_OUTPUTS / "h2_reference"
    h2_dir.mkdir(parents=True, exist_ok=True)
    energy = calculate_h2_molecule_energy(str(h2_dir))
    if energy is None:
        raise RuntimeError("Failed to compute H2 reference energy; aborting run")

    if world.rank == 0:          # cache zapisuje iba rank 0 (MPI-safe)
        H2_REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(H2_REFERENCE_FILE, 'w') as f:
            json.dump(
                {
                    'E_h2_eV': float(energy),
                    'h2_config_sha8': _h2_config_fingerprint(),
                    'grid_spacing_h_A': GPAW_H,
                    'timestamp': datetime.now().isoformat(),
                    # POZOR: musí to byť konfigurácia H₂, NIE GPAW_CONFIG slabu.
                    # Predtým sa tu dumpoval globálny GPAW_CONFIG, takže JSON hlásil
                    # kpts=[2,2,1] / relaxation_steps=8 — ani jedno nebola pravda o H₂.
                    'gpaw_config': {
                        'mode': GPAW_CONFIG['mode'],
                        'basis': GPAW_CONFIG['basis'],
                        'xc': GPAW_CONFIG['xc'],
                        'kpts': [1, 1, 1],
                        'periodic': False,
                        'box': 'atoms.center(vacuum=10) → 20 × 20 × (d+20) Å',
                        'occupations': 'zero-width (GPAW default, neperiodický systém)',
                        'convergence': GPAW_CONFIG['convergence'],
                    },
                    'relaxation': {'optimizer': 'BFGS', 'fmax_target_eV_A': 0.01,
                                   'steps_max': 50},
                    'h2_measured': dict(_H2_PROVENANCE),
                },
                f,
                indent=2,
            )
        print(f"✓ Saved H2 reference cache: {H2_REFERENCE_FILE}")
    return energy, "computed"


def calculate_surface_properties(formula, miller, slab_file, output_dir, e_h2, h2_source):
    """
    Calculate all energies and ΔGH for a surface.

    Loads the slab once and passes it to both clean and H calculations.
    Writes result to CSV incrementally.
    """
    
    print(f"\n{'='*60}")
    print(f"Calculating {formula} {miller}")
    print(f"{'='*60}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    result = {
        'formula': formula,
        'surface': miller,
        'timestamp': datetime.now().isoformat(),
        'status': 'pending',
        'E_clean_slab': None,
        'E_slab_with_h': None,
        'E_h2': None,
        'ΔGH': None,
        'adsorption_site': None,
        'h_position': None,
        'h2_source': h2_source,
        'relaxed': bool(USE_RELAXATION),
        'error': None,
    }
    
    try:
        # Check if file exists
        if not os.path.exists(slab_file):
            raise FileNotFoundError(f"POSCAR not found: {slab_file}")

        # Load slab ONCE
        slab = _prepare_slab(slab_file)
        
        # Calculate clean slab
        e_clean = calculate_clean_slab_energy(slab, output_dir)
        if e_clean is None:
            raise RuntimeError("Failed to calculate clean slab energy")
        result['E_clean_slab'] = float(e_clean)
        
        # Calculate slab with H (pass same slab object)
        e_with_h, best_site, best_position = calculate_slab_with_h_energy(slab, output_dir)
        if e_with_h is None:
            raise RuntimeError("Failed to calculate slab+H energy")
        result['E_slab_with_h'] = float(e_with_h)
        result['adsorption_site'] = best_site
        result['h_position'] = best_position
        
        result['E_h2'] = float(e_h2)
        
        # Calculate ΔGH
        # ΔGH = E(slab+H) - E(slab) - 0.5 * E(H2)
        delta_gh = e_with_h - e_clean - 0.5 * e_h2
        result['ΔGH'] = float(delta_gh)
        result['status'] = 'completed'
        
        print(f"\n  Results for {formula} {miller}:")
        print(f"    E(clean slab) = {e_clean:.6f} eV")
        print(f"    E(slab+H)     = {e_with_h:.6f} eV")
        print(f"    E(H₂)         = {e_h2:.6f} eV")
        print(f"    ΔGH           = {delta_gh:.6f} eV ← KEY RESULT!")
        
        if -0.2 < delta_gh < 0.2:
            print(f"    ✓ EXCELLENT! Near optimal for HER")
        elif -0.5 < delta_gh < 0.5:
            print(f"    ○ GOOD, reasonable for HER")
        else:
            print(f"    ⚠️  Outside typical HER range")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"  ✗ Error: {e}")

    # Write result to CSV immediately
    row = [
        result['formula'],
        result['surface'],
        'H',
        result['E_clean_slab'],
        result['E_slab_with_h'],
        result['E_h2'],
        result['ΔGH'],
        result['ΔGH'],  # descriptor_eV = ΔGH
        f'GPAW_{GPAW_CONFIG["xc"]}',
        result.get('adsorption_site'),
        result.get('h2_source'),
        result.get('relaxed'),
        result['status'],
        result['timestamp'],
    ]
    _append_result_csv(row, OUTPUT_CSV)

    return result


def _write_failed_row(formula, surface, h2_source, reason):
    """Write a failed result row directly (used for timeout/worker failures)."""
    row = [
        formula,
        surface,
        'H',
        None,
        None,
        None,
        None,
        None,
        'GPAW_LDA',
        None,
        h2_source,
        bool(USE_RELAXATION),
        'failed',
        datetime.now().isoformat(),
    ]
    _append_result_csv(row, OUTPUT_CSV)
    print(f"  ✗ Marked failed: {formula} {surface} ({reason})")


def _timeout_handler(signum, frame):
    raise TimeoutError("Per-structure timeout reached")


def _compute_one(args):
    """Worker function for parallel execution."""
    formula, surface, poscar_dir, e_h2, h2_source, max_seconds = args
    poscar_file = poscar_dir / "POSCAR"
    output_dir = GPAW_OUTPUTS / poscar_dir.name

    if max_seconds and max_seconds > 0:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(max_seconds))

    t0 = time.time()
    try:
        return calculate_surface_properties(
            formula=formula,
            miller=surface,
            slab_file=str(poscar_file),
            output_dir=str(output_dir),
            e_h2=e_h2,
            h2_source=h2_source,
        )
    except TimeoutError as exc:
        _write_failed_row(formula, surface, h2_source, str(exc))
        return {
            'formula': formula,
            'surface': surface,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(exc),
        }
    finally:
        elapsed_min = (time.time() - t0) / 60.0
        print(f"⏱ Finished worker task {formula} {surface} in {elapsed_min:.1f} min")
        if max_seconds and max_seconds > 0:
            signal.alarm(0)


# ── AdsorbML mode ───────────────────────────────────────────────
def _load_adsorbml_structures(csv_path):
    """Read ranked_candidates.csv and return list of dicts for _compute_one_adsorbml."""
    df = pd.read_csv(csv_path)
    required = {'slab_name', 'slab_file', 'candidate_file', 'gibbs_free_ml_eV'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ranked_candidates.csv missing columns: {missing}")
    return df.to_dict('records')


def _compute_one_adsorbml(args):
    """Worker for AdsorbML mode: single-point DFT on a pre-relaxed slab + adslab."""
    row, e_h2, h2_source = args
    slab_name    = row['slab_name']
    slab_file    = row['slab_file']
    adslab_file  = row['candidate_file']
    gibbs_ml     = row['gibbs_free_ml_eV']

    # Výpočtový adresár kľúčovaný AJ hashom konfigurácie. Predtým bol kľúčovaný len
    # menom, takže kampane s rôznym XC/k-mriežkou si navzájom prepisovali logy
    # (napr. Mo2C_(111)/clean_slab.txt bol z 27.7. ale clean_slab_relax.log z 24.7.)
    # → DFT čísla sa nedali auditovať. Podadresár to oddelí a staré dáta nechá byť.
    cfg8 = _config_fingerprint()
    output_dir = str(GPAW_OUTPUTS / slab_name / cfg8)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"AdsorbML DFT validation: {slab_name}")
    print(f"  ML ΔG*H = {gibbs_ml:.4f} eV")
    print(f"{'='*60}")

    timestamp = datetime.now().isoformat()
    status = 'failed'
    e_clean = e_with_h = dgh = None
    err_text = ''
    import traceback as _tb; _LAST_EXC['tb'] = _tb.format_exc()
    clean_slab = adslab = None

    try:
        clean_slab = _prepare_slab(slab_file)
        _apply_initial_magmoms(clean_slab)
        clean_slab.calc = setup_gpaw_calculator(label=f'{output_dir}/clean_slab')
        if ADSORBML_RELAX:
            _maybe_relax(clean_slab, f'{output_dir}/clean_slab')   # DFT relaxácia (RPBE)
        e_clean = clean_slab.get_potential_energy()
        print(f"     ✓ Clean slab: E = {e_clean:.6f} eV")

        adslab = _prepare_slab(adslab_file)
        # BRÁNA SYMETRIE: adslab musí byť clean + presne jeden H a v tej istej bunke.
        # Bez tejto kontroly sa dá ticho odčítať energia dvoch nekompatibilných
        # systémov (predtým malo 24 adresárov relax len na jednej strane).
        if len(adslab) != len(clean_slab) + 1:
            raise ValueError(
                f"nekompatibilný pár: adslab {len(adslab)} at. != clean {len(clean_slab)}+1")
        if adslab.get_chemical_symbols()[-1] != 'H':
            raise ValueError("posledný atóm adslabu nie je H")
        if not np.allclose(adslab.cell[:], clean_slab.cell[:], atol=1e-4):
            raise ValueError("adslab a clean slab majú rôznu bunku")

        _apply_initial_magmoms(adslab)
        adslab.calc = setup_gpaw_calculator(label=f'{output_dir}/adslab')
        if ADSORBML_RELAX:
            _maybe_relax(adslab, f'{output_dir}/adslab')           # DFT relaxácia (RPBE)
        e_with_h = adslab.get_potential_energy()
        print(f"     ✓ Adslab: E = {e_with_h:.6f} eV")

        dgh = e_with_h - e_clean - 0.5 * e_h2
        # `completed` znamená LEN že výpočet dobehol. O použiteľnosti rozhoduje
        # relax_converged_* / fmax_final_* v provenance stĺpcoch — konvergenciu
        # NEZAMIEŇAŤ s dokončením (to bola príčina rozstrelu E_clean 4.56 eV).
        cc = clean_slab.info.get('relax_converged')
        ca = adslab.info.get('relax_converged')
        status = 'completed' if (cc is not False and ca is not False) else 'completed_unconverged'
        print(f"  ΔGH (DFT) = {dgh:.6f} eV   ML = {gibbs_ml:.4f} eV   [{status}]")

    except Exception as exc:
        print(f"  ✗ Error: {exc}")
        err_text = f"{type(exc).__name__}: {exc}"
        import traceback as _tb; _LAST_EXC['tb'] = _tb.format_exc()

    row_data = [
        slab_name, slab_name, 'H',
        e_clean, e_with_h, e_h2,
        dgh, dgh, f'GPAW_{GPAW_CONFIG["xc"]}_adsorbml',
        # PROVENANCE FIX: tu bol hardcoded literál `False`, takže CSV tvrdilo
        # relaxed=False AJ KEĎ ADSORBML_RELAX=1 a relaxácia reálne prebehla
        # (dokázateľné z *_relax.log: BFGS kroky, fmax). Pole klamalo o metóde.
        'adsorbml_best', h2_source, bool(ADSORBML_RELAX),
        status, timestamp, gibbs_ml,
    ]

    def _inf(a, k, default=''):
        return a.info.get(k, default) if a is not None else default

    row_data += [
        _inf(clean_slab, 'relax_converged'), _inf(clean_slab, 'fmax_final'),
        _inf(clean_slab, 'relax_steps'),
        _inf(adslab, 'relax_converged'), _inf(adslab, 'fmax_final'),
        _inf(adslab, 'relax_steps'),
        GPAW_CONFIG['xc'], GPAW_CONFIG['mode'], GPAW_CONFIG.get('basis', ''),
        'x'.join(str(k) for k in GPAW_CONFIG['kpts']),
        GPAW_SIGMA or '0.1(default)', GPAW_SYMMETRY or 'on',
        RELAXATION_CONFIG['fmax'], RELAXATION_CONFIG['steps'], SPINPOL,
        len(clean_slab) if clean_slab is not None else '',
        len(adslab) if adslab is not None else '',
        output_dir, cfg8,
    ]
    if err_text:
        print(f"     (chyba zaznamenaná: {err_text})")
        # Traceback, nie len text. Bez neho sa numpy chyby typu
        # "axes don't match array" nedajú lokalizovať a debug sa robí naslepo.
        if _LAST_EXC.get('tb'):
            print(_LAST_EXC['tb'], flush=True)
    _append_result_csv(row_data, ADSORBML_OUTPUT_CSV)
    return {'formula': slab_name, 'surface': slab_name, 'status': status,
            'ΔGH': dgh, 'gibbs_free_ml_eV': gibbs_ml, 'timestamp': timestamp}


def run_adsorbml_calculations(candidates_csv, workers_override=None, selected_names=None):
    """Run single-point GPAW on AdsorbML pre-relaxed structures."""
    print("\n" + "="*60)
    print("GPAW AdsorbML Validation Mode")
    print("="*60)
    print(f"Starting time: {datetime.now()}")
    print(f"Candidates CSV: {candidates_csv}")

    _ensure_csv_header(ADSORBML_OUTPUT_CSV, columns=ADSORBML_CSV_COLUMNS)

    completed = _load_completed_keys(ADSORBML_OUTPUT_CSV)
    if completed:
        print(f"✓ Checkpoint: {len(completed)} structures already completed, will skip")

    e_h2, h2_source = get_h2_reference_energy()

    structures = _load_adsorbml_structures(candidates_csv)
    if selected_names:
        selected_set = set(selected_names)
        structures = [s for s in structures if s['slab_name'] in selected_set]

    pending = [
        (row, e_h2, h2_source)
        for row in structures
        if (row['slab_name'], row['slab_name']) not in completed
    ]
    print(f"Structures to compute: {len(pending)} (skipped {len(structures) - len(pending)} completed)")

    if not pending:
        print("✓ All structures already completed!")
        return []

    max_workers = _detect_max_workers(override_workers=workers_override)
    if world.size > 1:           # pod MPI: 1 štruktúra/proces, MPI robí paralelizmus
        max_workers = 1
    all_results = []

    if max_workers <= 1:
        for args in pending:
            if _shutdown_requested:
                break
            all_results.append(_compute_one_adsorbml(args))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_compute_one_adsorbml, args): args for args in pending}
            for future in as_completed(future_map):
                if _shutdown_requested:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    all_results.append(future.result())
                except Exception as exc:
                    args = future_map[future]
                    print(f"  ✗ Worker exception for {args[0]['slab_name']}: {exc}")

    return all_results


def run_calculations_parallel(formulas=['MoS2', 'MoSe2', 'MoP', 'Mo2N'],
                             millers=['(100)', '(110)', '(111)'],
                             base_dir=None,
                             results_file=None,
                             use_discovery=True,
                             include_patterns=None,
                             selected_structure_names=None,
                             workers_override=None,
                             max_hours_per_structure=None):
    """
    Run all calculations with parallel workers and checkpoint/resume.

    Args:
        include_patterns: optional list of glob patterns to filter structures
    """
    
    print("\n" + "="*60)
    print("GPAW H Adsorption Energy Calculator (v2)")
    print("="*60)
    print(f"\nStarting time: {datetime.now()}")
    print(f"Relaxation: {'ON (' + str(RELAXATION_CONFIG['steps']) + ' steps)' if USE_RELAXATION else 'OFF'}")
    if include_patterns:
        print(f"Include filter: {', '.join(include_patterns)}")
    if max_hours_per_structure and max_hours_per_structure > 0:
        print(f"Per-structure timeout: {max_hours_per_structure:.2f} hours")

    base_dir = Path(base_dir) if base_dir else DATA_INPUTS

    # Ensure output CSV exists with header
    _ensure_csv_header(OUTPUT_CSV)

    # Load checkpoint: skip already-completed structures
    completed = _load_completed_keys(OUTPUT_CSV)
    if completed:
        print(f"✓ Checkpoint: {len(completed)} structures already completed, will skip")

    e_h2, h2_source = get_h2_reference_energy()
    
    all_results = []
    
    if use_discovery:
        structures = discover_structures(base_dir, include_patterns=include_patterns)
        structures = filter_structures_by_name(structures, selected_structure_names)
        if not structures:
            print("\n⚠️  No POSCAR files found in data/inputs/VASP_inputs")
            return all_results

        # Filter out already-completed
        pending = []
        max_seconds = 0
        if max_hours_per_structure and max_hours_per_structure > 0:
            max_seconds = int(max_hours_per_structure * 3600)
        for formula, surface, poscar_dir in structures:
            if (formula, surface) in completed:
                continue
            pending.append((formula, surface, poscar_dir, e_h2, h2_source, max_seconds))

        print(f"Structures to compute: {len(pending)} (skipped {len(structures) - len(pending)} completed)")

        if not pending:
            print("✓ All structures already completed!")
            return all_results

        max_workers = _detect_max_workers(override_workers=workers_override)

        if max_workers <= 1:
            # Sequential fallback
            for args in pending:
                if _shutdown_requested:
                    print("⚠️  Shutdown: stopping before next structure")
                    break
                result = _compute_one(args)
                all_results.append(result)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_compute_one, args): args for args in pending}

                for future in as_completed(future_map):
                    if _shutdown_requested:
                        print("⚠️  Shutdown: cancelling remaining futures")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as exc:
                        args = future_map[future]
                        print(f"  ✗ Worker exception for {args[0]} {args[1]}: {exc}")
    else:
        for formula in formulas:
            for miller in millers:
                if _shutdown_requested:
                    break
                if (formula, miller) in completed:
                    continue
                dir_name = f"{formula}_{miller.replace('(', '').replace(')', '')}"
                poscar_dir = Path(base_dir) / f"{formula}_{miller}"
                poscar_file = poscar_dir / "POSCAR"
                output_dir = GPAW_OUTPUTS / dir_name
                
                result = calculate_surface_properties(
                    formula=formula,
                    miller=miller,
                    slab_file=str(poscar_file),
                    output_dir=str(output_dir),
                    e_h2=e_h2,
                    h2_source=h2_source,
                )
                all_results.append(result)
    
    return all_results


def save_results(results, json_file=None,
                csv_file=None):
    """
    Save summary JSON and print statistics.

    Note: CSV is written incrementally during computation.
    This function saves the JSON backup and prints a summary.
    """
    
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    json_file = Path(json_file) if json_file else OUTPUT_JSON
    csv_file = Path(csv_file) if csv_file else OUTPUT_CSV

    # Save JSON (full details)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Full results saved to: {json_file}")
    print(f"✓ Incremental CSV at: {csv_file}")
    
    # Print summary from the CSV (which has all results including previous runs)
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        df = pd.DataFrame()

    if len(df) > 0:
        completed = df[df['status'] == 'completed']
        print(f"\nTotal entries in CSV: {len(df)}")
        print(f"Successfully calculated: {len(completed)} surfaces")

        if len(completed) > 0:
            print("\nΔGH statistics by formula:")
            for formula in sorted(completed['formula'].unique()):
                formula_df = completed[completed['formula'] == formula]
                dgh_values = formula_df['ΔGH_eV'].dropna().values
                if len(dgh_values) > 0:
                    print(f"  {formula}:")
                    print(f"    Count: {len(dgh_values)}")
                    print(f"    Mean ΔGH: {np.mean(dgh_values):.4f} eV")
                    print(f"    Min ΔGH:  {np.min(dgh_values):.4f} eV (best surface)")
                    print(f"    Max ΔGH:  {np.max(dgh_values):.4f} eV")

            # Highlight best candidates
            excellent = completed[completed['ΔGH_eV'].abs() < 0.2]
            if len(excellent) > 0:
                print(f"\n✓ EXCELLENT candidates (|ΔGH| < 0.2 eV): {len(excellent)}")
                print(excellent[['formula', 'surface_facet', 'ΔGH_eV']].to_string(index=False))

        failed = df[df['status'] == 'failed']
        if len(failed) > 0:
            print(f"\n⚠️  Failed calculations: {len(failed)}")
            print(failed[['formula', 'surface_facet', 'status']].to_string(index=False))
    
    print(f"\nEnd time: {datetime.now()}")


# ── Pre-defined machine splits ───────────────────────────────────
MACHINE_SPLITS = {
    # DEVANA (64 CPUs): 159 structures (~46.5%)
    'devana': [
        'Ni_Mo2N_interface_*',
        'Ni_MoS2_interface_*',
        'Mo2N_*',
        'MoS2_*',
        'MoSe2_*',
        'MoP_*',
        'graphene_*',
        'Ni2_on_*',
        'Ni4_on_*',
    ],
    # NODE1 (32 CPUs): 79 structures (~23.1%)
    'node1': [
        'Ni_MoB_interface_*',
        'MoB_*',
    ],
    # NODE2 (24 CPUs): 61 structures (~17.8%)
    'node2': [
        'Ni_Mo2C_interface_*',
        'Ni_Ti3C2O2_interface_*',
        'Ti3C2O2_*',
    ],
    # NODE3 (16 CPUs): 43 structures (~12.6%)
    'node3': [
        'Mo2C_*',
    ],
}


def main():
    """Main execution"""

    global USE_RELAXATION, CORES_PER_CALC

    parser = argparse.ArgumentParser(
        description='GPAW H Adsorption Calculator (v2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Machine splits (use --machine instead of --include):\n'
            + '\n'.join(f'  {k:14s} → {" , ".join(v)}' for k, v in MACHINE_SPLITS.items())
        ),
    )
    parser.add_argument(
        '--include', type=str, default=None,
        help='Comma-separated glob patterns to filter structures, '
             'e.g. "Ni_Mo2C_interface_*,Ni_MoB_interface_*"',
    )
    parser.add_argument(
        '--machine', type=str, default=None,
        choices=list(MACHINE_SPLITS.keys()),
        help='Use a pre-defined split for a specific machine',
    )
    parser.add_argument(
        '--list', action='store_true', dest='list_only',
        help='Only list structures that would be computed, then exit',
    )
    parser.add_argument(
        '--write-structure-list', type=str, default=None,
        help='Write matched structure directory names to a text file, one per line, then exit',
    )
    parser.add_argument(
        '--structure-name', action='append', default=None,
        help='Run only the given structure directory name. May be provided multiple times or as a comma-separated list.',
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Manual number of parallel workers (overrides auto-detection)',
    )
    parser.add_argument(
        '--cores-per-calc', type=int, default=None,
        help='CPU cores budget per calculation used by auto worker detection',
    )
    parser.add_argument(
        '--relax-steps', type=int, default=None,
        help='Override relaxation steps (default from config)',
    )
    parser.add_argument(
        '--fmax', type=float, default=None,
        help='Override BFGS force threshold in eV/A',
    )
    parser.add_argument(
        '--no-relax', action='store_true',
        help='Disable structural relaxation (single-point energies only)',
    )
    parser.add_argument(
        '--kpts', type=str, default=None,
        help='Override k-point mesh as comma-separated triple, e.g. 2,2,1',
    )
    parser.add_argument(
        '--max-hours-per-structure', type=float, default=0.0,
        help='Hard timeout per structure in hours (0 disables timeout)',
    )
    parser.add_argument(
        '--adsorbml-candidates', type=str, default=None,
        metavar='PATH',
        help='Path to ranked_candidates.csv from scripts/adsorbml/3-extract_rank.py. '
             'When provided, skips POSCAR discovery and H-site search and instead '
             'runs single-point DFT on AdsorbML pre-relaxed structures. '
             'Results are written to data/outputs/gpaw_adsorbml_results.csv.',
    )
    args = parser.parse_args()

    # Resolve include patterns
    include_patterns = None
    if args.machine:
        include_patterns = MACHINE_SPLITS[args.machine]
        print(f"Using machine split: {args.machine}")
    elif args.include:
        include_patterns = [p.strip() for p in args.include.split(',')]

    selected_structure_names = None
    if args.structure_name:
        selected_structure_names = []
        for value in args.structure_name:
            selected_structure_names.extend(part.strip() for part in value.split(',') if part.strip())
        print(f"Explicit structures: {', '.join(selected_structure_names)}")

    # Runtime overrides
    if args.cores_per_calc is not None:
        CORES_PER_CALC = max(1, int(args.cores_per_calc))
        print(f"Override: CORES_PER_CALC={CORES_PER_CALC}")

    _set_thread_env(CORES_PER_CALC)

    if args.no_relax:
        USE_RELAXATION = False
        print("Override: relaxation disabled")

    if args.relax_steps is not None:
        RELAXATION_CONFIG['steps'] = max(0, int(args.relax_steps))
        print(f"Override: relax_steps={RELAXATION_CONFIG['steps']}")

    if args.fmax is not None:
        RELAXATION_CONFIG['fmax'] = float(args.fmax)
        print(f"Override: fmax={RELAXATION_CONFIG['fmax']}")

    if args.kpts:
        try:
            parts = tuple(int(x.strip()) for x in args.kpts.split(','))
            if len(parts) != 3:
                raise ValueError("kpts must have 3 integers")
            GPAW_CONFIG['kpts'] = parts
            print(f"Override: kpts={GPAW_CONFIG['kpts']}")
        except Exception as exc:
            raise ValueError(f"Invalid --kpts '{args.kpts}': {exc}")

    # List-only mode
    if args.list_only or args.write_structure_list:
        structures = discover_structures(DATA_INPUTS, include_patterns=include_patterns)
        structures = filter_structures_by_name(structures, selected_structure_names)
        print(f"\nStructures matched: {len(structures)}")
        for formula, surface, poscar_dir in structures:
            print(f"  {poscar_dir.name}")
        if args.write_structure_list:
            output_path = Path(args.write_structure_list)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(''.join(f"{poscar_dir.name}\n" for _, _, poscar_dir in structures))
            print(f"\n✓ Wrote structure manifest: {output_path}")
        if args.list_only or args.write_structure_list:
            return

    if selected_structure_names:
        args.workers = 1
        print("Single-structure mode: forcing workers=1 for scheduler-friendly execution")

    # H₂ cache sa už NEinvaliduje mazaním — porovnanie fingerprintu je vnútri
    # get_h2_reference_energy(). Starý blok mazal súbor pri nesúlade dvoch kľúčov,
    # čo v array jobe znamenalo 47 taskov mažúcich ten istý súbor naraz (race),
    # a keďže jeden z tých kľúčov ('relaxation_steps') sa medzitým zo schémy
    # stratil, invalidácia sa spúšťala VŽDY (job 74571: 47/47 FAILED).
    
    # ── AdsorbML validation mode ─────────────────────────────────
    if args.adsorbml_candidates:
        candidates_path = Path(args.adsorbml_candidates)
        if not candidates_path.exists():
            raise FileNotFoundError(f"--adsorbml-candidates file not found: {candidates_path}")
        results = run_adsorbml_calculations(
            candidates_csv=str(candidates_path),
            workers_override=args.workers,
            selected_names=selected_structure_names,
        )
        save_results(results, csv_file=ADSORBML_OUTPUT_CSV)
        print("\n✓ Done! AdsorbML DFT results saved to:")
        print(f"  - {ADSORBML_OUTPUT_CSV}")
        return

    # ── Standard fresh-POSCAR mode ───────────────────────────────
    results = run_calculations_parallel(
        include_patterns=include_patterns,
        selected_structure_names=selected_structure_names,
        workers_override=args.workers,
        max_hours_per_structure=args.max_hours_per_structure,
    )

    # Save JSON summary
    save_results(results)

    print("\n✓ Done! Results saved to:")
    print(f"  - {OUTPUT_CSV} (incremental, crash-safe)")
    print(f"  - {OUTPUT_JSON} (JSON summary)")
    print("\nNext step: Combine with OCx24 and train ML model!")


if __name__ == '__main__':
    main()
