#!/usr/bin/env python3
"""monitor_camp.py — production monitor for the DFT campaign (UMA ΔG_H validation).

Fourth of the family:
    monitor_uma.sh   →  UMA/AdsorbML on GPU
    monitor_val.sh   →  convergence validations (k-mesh, h, σ)
    monitor_dft.sh   →  the earlier DFT run (gpaw_calculations)
    monitor_camp.py  →  THIS campaign (campaign_dft) + crystallography

Why Python rather than bash+awk like the others: surface termination,
supercell, adsorption site and coverage cannot be derived from a filename —
they are computed from atomic positions, the cell and the FixAtoms mask
through ASE. The crystallography is cached in campaign_meta.json.

ADAPTIVE WIDTH: at 172 columns or more the crystallography gets its own
columns; below that it folds into a quiet sub-row under each structure. The
label column is a fixed 11 characters, so values line up exactly.

RUNNING ΔG_H: as soon as both the clean and the adslab side have started,
~ΔG_H = E_adslab − E_clean − ½E(H₂) + ZPE is formed from the latest BFGS
energies. It is provisional — neither side need be finished — hence the tilde.
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("REPO_ROOT",
                           Path(__file__).resolve().parents[1]))
CDIR = REPO / "data/outputs/campaign_dft"
CLEAN = REPO / "data/uma_relaxed_campaign"
RANKED = REPO / "data/adsorbml_results/ranked_candidates_campaign.csv"
CSV_MAIN = Path(os.environ.get("CAMPAIGN_CSV", str(CDIR / "campaign_dft_results.csv")))
META = CDIR / "campaign_meta.json"

TOTAL = int(os.environ.get("CAMPAIGN_TOTAL", "83"))

# Cluster environment — override via env when running under another account.
SLURM_USER    = os.environ.get("SLURM_USER", os.environ.get("USER", ""))
SLURM_ACCOUNT = os.environ.get("SLURM_ACCOUNT", "")
SINCE         = os.environ.get("SACCT_SINCE", "2026-07-29")
SINCE_PROJECT = os.environ.get("SACCT_SINCE_PROJECT", "2026-01-01")
META_VER = 2          # bump when the field format changes; forces a cache rebuild
E_H2 = -6.466266559623192          # h=0.16, RPBE/dzp, relaxed d(H–H) = 0.7755 Å
ZPE = 0.24
# The allocation sits on the high-memory QOS: GrpTRESMins billing = 48,000,000
# minutes / 60 = 800,000 core-hours, verified with `sacctmgr show qos <qos>`.
# The generic CPU QOS has billing=0, so cpu_long/cpu_short are closed to the
# project and the whole budget lives in those 800,000 core-hours.
FAT_QUOTA = int(os.environ.get("FAT_QUOTA", "800000"))   # core-h on the high-memory partition
NODES_TOT, PART_CORES = 15, 4800   # cn046-060, cpu_hm_*

try:
    TERM = int(os.environ.get("COLS") or shutil.get_terminal_size((120, 40)).columns)
except Exception:
    TERM = 120
W = max(96, min(TERM, 210))
WIDE = W >= 172

C = dict(r="\033[0m", d="\033[2m", b="\033[1m", g="\033[32m", y="\033[33m",
         R="\033[31m", c="\033[36m", m="\033[35m")
if os.environ.get("NO_COLOR") or (not sys.stdout.isatty()
                                  and os.environ.get("MONITOR_COLOR") != "1"):
    C = {k: "" for k in C}

# Space groups from the corrected bulk builders in
# scripts/generate_structures.py, each carrying its literature citation.
PHASE = {
    "MoP":     dict(f="MoP",     proto="WC type (Bₕ)",        sg="P-6m2",   n=187, sys="hex", anion="P"),
    "MoS2":    dict(f="MoS₂",    proto="2H polytype",         sg="P6₃/mmc", n=194, sys="hex", anion="S"),
    "MoSe2":   dict(f="MoSe₂",   proto="2H polytype",         sg="P6₃/mmc", n=194, sys="hex", anion="Se"),
    "Mo2C":    dict(f="Mo₂C",    proto="orthorh. ζ-Fe₂N",     sg="Pbcn",    n=60,  sys="ort", anion="C"),
    "Mo2N":    dict(f="γ-Mo₂N",  proto="defect rock salt",    sg="Fm-3m",   n=225, sys="cub", anion="N"),
    "MoB":     dict(f="MoB",     proto="CrB type",            sg="Cmcm",    n=63,  sys="ort", anion="B"),
    "Ti3C2O2": dict(f="Ti₃C₂O₂", proto="MXene, O-term.",      sg="P-3m1",   n=164, sys="tri", anion="O"),
}


def sh(cmd, t=25):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=t).stdout
    except Exception:
        return ""


def cut(x, n):
    x = "" if x is None else str(x)
    return x if len(x) <= n else x[:n - 1] + "…"


def pad(x, n, col=""):
    """Pad to an exact width. Colour is applied only after the length is
    computed, otherwise the ANSI codes would break the column alignment."""
    x = cut(x, n)
    return (f"{col}{x}{C['r']}" if col else x) + " " * (n - len(x))



def conv_pct(fm, target=0.05):
    """Relaxation progress in %, measured on a LOGARITHMIC fmax scale.

    The number of remaining steps cannot be known in advance, but the distance
    to the target can: fmax decays roughly exponentially, so
    log(f0/f)/log(f0/target) is a monotonic estimate of how much of the way is
    behind us. 0 % = at the start, 100 % = fmax <= target. This is convergence
    distance, not elapsed time.
    """
    if not fm:
        return None
    import math
    f0, f = max(fm[0], target * 1.001), max(fm[-1], 1e-6)
    if f <= target:
        return 100
    if f0 <= target:
        return 100
    return int(max(0, min(99, 100 * math.log(f0 / f) / math.log(f0 / target))))


def bar(pct, n=5):
    if pct is None:
        return " " * n
    fl = int(round(pct / 100 * n))
    return "▓" * fl + "░" * (n - fl)


# ── crystallography from the actual geometry ────────────────────────────────
def analyse(name, row):
    import numpy as np
    from ase.io import read
    fam, var = row["family"], row["variant"]
    ph = PHASE.get(fam, {})
    d = dict(name=name, family=fam, facet=row["facet"], variant=var,
             formula=ph.get("f", fam), proto=ph.get("proto", "?"), sg=ph.get("sg", "?"),
             sgn=ph.get("n", 0), lattice=ph.get("sys", "?"), anion=ph.get("anion", "X"))
    cf, af = CLEAN / f"{name}.traj", Path(row["candidate_file"])
    if not cf.is_file():
        return d
    slab = read(str(cf))
    z = slab.positions[:, 2]
    cell = slab.get_cell()
    d.update(n_atoms=len(slab),
             a_A=float(np.linalg.norm(cell[0])), b_A=float(np.linalg.norm(cell[1])),
             c_A=float(np.linalg.norm(cell[2])),
             thickness_A=float(z.max() - z.min()))
    d["vacuum_A"] = d["c_A"] - d["thickness_A"]
    # termination = species within 1 Å of the topmost / bottommost atom
    top = [slab[i].symbol for i in range(len(slab)) if z[i] > z.max() - 1.0]
    bot = [slab[i].symbol for i in range(len(slab)) if z[i] < z.min() + 1.0]
    d.update(term_top="+".join(sorted(set(top))), term_bot="+".join(sorted(set(bot))),
             n_top=len(top), symmetric=(set(top) == set(bot)))
    fixed = set()
    for c in slab.constraints:
        idx = getattr(c, "index", None)
        if idx is not None:
            fixed.update(int(i) for i in np.asarray(idx).ravel())
    d["n_fixed"] = len(fixed)
    an = d["anion"]
    d["defect"] = {"base": "—", f"vac{an}": f"1× vac {an}", f"vac2{an}": f"2× vac {an}",
                   "dopPt": "subst Pt"}.get(var, var)
    d["stoich"] = ("stoich." if var == "base"
                   else f"red. ({an}-def.)" if var.startswith("vac") else "doped")
    d.update(ads="—", site="—", coverage="—", h_height_A=None)
    if af.is_file():
        ads = read(str(af))
        if len(ads) == len(slab) + 1 and ads.get_chemical_symbols()[-1] == "H":
            hp, sub = ads.positions[-1], ads.positions[:-1]
            dist = np.linalg.norm(sub - hp, axis=1)
            dmin = float(dist.min())
            near = int((dist < dmin * 1.25).sum())   # H coordination = site type
            d.update(ads="H chemisorption", nn=near, d_HM_A=dmin,
                     nn_sym=ads[int(np.argmin(dist))].symbol,
                     h_height_A=float(hp[2] - z.max()),
                     site={1: "top", 2: "brdg", 3: "holl3", 4: "holl4"}.get(near, f"{near}f"))
            if d["n_top"]:
                d["coverage"] = f"1/{d['n_top']}"
    return d


def build_meta():
    if not RANKED.is_file():
        # A fresh checkout has no campaign data — say so plainly instead of
        # raising FileNotFoundError out of the render loop.
        print(f"No campaign data found.\n  expected: {RANKED}\n"
              f"  set REPO_ROOT=/path/to/repo to point the monitor elsewhere.",
              file=sys.stderr)
        raise SystemExit(1)
    rows = {r["slab_name"]: r for r in csv.DictReader(open(RANKED))}
    out = {}
    if META.is_file():
        try:
            out = json.loads(META.read_text())
        except Exception:
            out = {}
    ch = False
    for nm, r in rows.items():
        # The cache key must include the CANDIDATE. After the bond-aware
        # re-ranking the candidate_file changed, and without this the monitor
        # kept showing the old (unbound) adsorption geometry, i.e. it claimed
        # the fix had not taken effect.
        cand_key = Path(r["candidate_file"]).name
        if (nm not in out or out[nm].get("_v") != META_VER
                or out[nm].get("_cand") != cand_key):
            try:
                out[nm] = analyse(nm, r)
            except Exception as e:
                out[nm] = dict(name=nm, family=r["family"], facet=r["facet"],
                               variant=r["variant"], error=str(e)[:50])
            out[nm]["_v"] = META_VER
            out[nm]["_cand"] = cand_key
            ch = True
    if ch:
        META.parent.mkdir(parents=True, exist_ok=True)
        META.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out, rows


# ── live state of a single structure ────────────────────────────────────────
def live(name):
    st = {}
    for side in ("clean", "adslab"):
        s = {}
        lg = CDIR / f"{name}_{side}_relax.log"
        if lg.is_file():
            fm = []
            for ln in lg.read_text(errors="ignore").splitlines():
                m = re.match(r"^BFGS:\s+(\d+)\s+\S+\s+(\S+)\s+(\S+)", ln)
                if m:
                    try:
                        s["E"] = float(m.group(2)); fm.append(float(m.group(3)))
                    except ValueError:
                        pass
            # STEP COUNT = number of written lines, NOT the index of the last
            # one. After a restart ASE renumbers from 0 and appends to the same
            # log, so the last index reported "0/200" even with 18 steps
            # already done (seen on MoS2_(110)_vacS after the candidate fix).
            s["step"] = len(fm)
            s["restarts"] = max(0, lg.read_text(errors="ignore").count("Step     Time") - 1)
            s["fmax"] = fm[-1] if fm else None
            s["trend"] = fm[-3:]
            s["hist"] = fm            # full history → convergence progress
        t = CDIR / f"{name}_{side}.txt"
        if t.is_file():
            s["mtime"] = t.stat().st_mtime
            tail = t.read_text(errors="ignore")[-6000:]
            # ([^|]+) rather than (\S+): GPAW appends the convergence marker
            # 'c' directly to the number without a space ("-783.723917c|"), so
            # \S+ swallowed the pipe as well and float() failed, leaving E
            # displayed as "—".
            it = re.findall(r"^\|iter:\s*(\d+)\|[^|]*\|([^|]+)", tail, re.M)
            if it:
                s["scf"] = int(it[-1][0])
                # Take the energy from the LAST parsable line. At the start of
                # an SCF cycle GPAW writes a line with an EMPTY energy field
                # ("|iter: 28| 13:24:30 |          |"), so blindly taking it[-1]
                # gave E = "—" long after the SCF had started (γ-Mo₂N, MoB).
                for _, raw in reversed(it):
                    try:
                        s["scf_E"] = float(raw.strip().rstrip("c").strip())
                        break
                    except ValueError:
                        continue
            s["done"] = "Free energy" in tail
            try:    # the IBZ count is in the log HEADER, not the tail
                with open(t, errors="ignore") as fh:
                    g = re.search(r"Number of IBZ points:\s*(\d+)", fh.read(20000))
                if g:
                    s["ibz"] = int(g.group(1))
            except OSError:
                pass
        if s:
            st[side] = s
    return st


def render():
    meta, rows = build_meta()
    res = {}
    if CSV_MAIN.is_file():
        try:
            res = {r["slab_name"]: r for r in csv.DictReader(open(CSV_MAIN))}
        except Exception:
            pass
    resmap_dbg = res
    ok = {k: v for k, v in res.items() if v.get("status") == "ok"}
    err = {k: v for k, v in res.items() if v.get("status", "").startswith("error")}

    # ── SLURM ────────────────────────────────────────────────────────────────
    q = sh(f"squeue -u {SLURM_USER} -h -o '%i|%F|%K|%j|%T|%C|%M|%N'")
    run, pend, cores, nodes, arrays, oldest = [], 0, 0, set(), set(), 0.0
    for ln in q.splitlines():
        p = ln.split("|")
        # "camp*" = array jobs of the original campaign, "c-<slab>" =
        # per-structure jobs after the switch to configuration B, "np-" =
        # scaling tests. Without "c-" the monitor reported 1 job on 144 cores
        # while 32 jobs were actually running on 4176.
        # "s0-" and "strain-" are follow-up single-points submitted after the
        # campaign finished; they recompute ALREADY COMPLETED structures for a
        # different quantity, so they can appear both among the active runs and
        # as "✓ done" in the main table.
        if len(p) < 8 or not p[3].startswith(("camp", "c-", "np-", "s0-", "strain")):
            continue
        if p[4] == "RUNNING":
            run.append(p)
            cores += int(p[5] or 0)
            nodes.update(x for x in p[7].replace("[", " ").replace("]", " ")
                         .replace(",", " ").split() if x.startswith("cn"))
            arrays.add(p[1])
            # SLURM elapsed is [D-]HH:MM:SS. Days are DAYS, not another base-60
            # unit — the original code applied `sec = sec*60 + v` to the day
            # field as well, turning "1-01:56:15" (25.94 h) into 61.94 h.
            # Verified: 61.9375 h buggy vs 25.9375 h correct.
            try:
                el = p[6]
                days = 0
                if "-" in el:
                    dstr, el = el.split("-", 1)
                    days = int(dstr)
                fields = [float(v) for v in el.split(":")]
                while len(fields) < 3:
                    fields.insert(0, 0.0)          # M:S → 0:M:S
                h, m, sc = fields[-3], fields[-2], fields[-1]
                sec = days * 86400 + h * 3600 + m * 60 + sc
                oldest = max(oldest, sec / 3600)
            except (ValueError, IndexError):
                pass
        elif p[4] == "PENDING":
            pend += 1
    # The high-memory partition is billed against its own QOS (48,000,000
    # billing-minutes = 800,000 core-hours), NOT against the account: the
    # association leaves GrpTRESMins empty. The generic CPU QOS has an
    # allocation of 0, so cpu_long/cpu_short are unavailable and the entire
    # budget lives in those 800,000. Querying the full history takes 0.03 s.
    fat = sh(f"sacct -u {SLURM_USER} -S {SINCE} -X -n -o NCPUS,ElapsedRaw,AllocTRES%40 "
             "--parsable2 | awk -F'|' '$3 !~ /gres.gpu/ {t+=$1*$2} END{printf \"%.0f\", t/3600}'")
    fat_used = int(fat) if (fat or "").strip().isdigit() else 0
    fp = sh(f"sacct -A {SLURM_ACCOUNT} -S {SINCE_PROJECT} -X -n -P -o NCPUS,ElapsedRaw,Partition "
            "| awk -F'|' '$3 ~ /hm/ {t+=$1*$2} END{printf \"%.0f\", t/3600}'")
    fat_proj = int(fp) if (fp or "").strip().isdigit() else 0
    # ETA must NOT be remaining/rate — that assumes sequential execution,
    # while most of the remaining structures run CONCURRENTLY and finish at
    # roughly the same time. The correct model is wave-based: how many waves of
    # `concurrency` structures are still needed, times the mean wall time per
    # structure taken from the finished ones (wall_s column).
    walls = []
    for r_ in ok.values():
        try:
            walls.append(float(r_["wall_s"]) / 3600)
        except (KeyError, ValueError, TypeError):
            pass
    mean_wall = sum(walls) / len(walls) if walls else None
    remain = TOTAL - len(ok)
    conc = max(1, len(run))
    # rate must NOT be done/age_of_oldest_job — after a campaign restart the
    # jobs are young while many structures are already finished, which produced
    # a nonsensical "36.7/h, ETA 205 h". Throughput is conc/mean_wall: `conc`
    # structures in parallel, each taking mean_wall hours.
    rate = (conc / mean_wall) if mean_wall else 0.0
    if mean_wall and remain > 0:
        waves = -(-remain // conc)                     # ceil
        eta = waves * mean_wall
    else:
        eta = None

    out = ["\033[H\033[2J"]
    A = out.append
    A(f"{C['b']}╔{'═'*(W-2)}╗{C['r']}")
    t1 = "DFT CAMPAIGN ⇄ HPC   AdsorbML/UMA ΔG_H validation"
    t2 = f"{len(run)} jobs · {cores} cores · {time.strftime('%H:%M:%S')}"
    A(f"{C['b']}║{C['r']} {t1:<{max(1, W-5-len(t2))}}{C['d']}{t2}{C['r']} {C['b']}║{C['r']}")
    A(f"{C['b']}╚{'═'*(W-2)}╝{C['r']}")

    LB = 11        # fixed label width → every value starts in the same column
    bn = 24
    fl = int(len(ok) / TOTAL * bn)
    # "since the run started" was misleading: `oldest` is the age of the
    # OLDEST RUNNING job, which resets on every campaign restart. Named honestly.
    _ids = sorted(arrays, key=lambda s: (len(s), s))
    if len(_ids) > 6:                    # 27+ per-structure jobs would not fit
        _idtxt = f"{len(_ids)}× ({_ids[0]}…{_ids[-1]})"
    else:
        _idtxt = " ".join(_ids) or "—"
    A(f" {C['d']}{'jobs':<{LB}}{_idtxt:<44}"
      f"oldest running job {oldest:.2f} h{C['r']}")
    A("")
    A(f" {C['b']}{'Progress':<{LB}}{C['r']}[{C['g']}{'█'*fl}{C['r']}{'░'*(bn-fl)}] "
      f"{C['b']}{len(ok)}/{TOTAL}{C['r']} ({100*len(ok)/TOTAL:.0f} %)"
      f"    rate {rate:.1f}/h    ETA {f'{eta:.1f} h' if eta else '—'}"
      f"   {C['d']}({f'{mean_wall:.2f} h/struct' if mean_wall else '—'}, "
      f"{-(-remain // conc) if mean_wall else '?'} waves × {conc} concurrent){C['r']}")
    A(f" {C['b']}{'Queue':<{LB}}{C['r']}running {C['g']}{len(run)}{C['r']} · pending {pend} · "
      f"done {C['g']}{len(ok)}{C['r']} · failed "
      f"{C['R'] if err else ''}{len(err)}{C['r']} · remaining {TOTAL-len(ok)}")
    A(f" {C['b']}{'Resources':<{LB}}{C['r']}{C['c']}{cores}{C['r']} of {PART_CORES} high-mem cores "
      f"({100*cores/PART_CORES:.0f} %) · {C['c']}{len(nodes)}{C['r']}/{NODES_TOT} nodes · "
      f"{cores} threads (OMP=1, ThreadsPerCore=1) · 3500 MB/core")
    _pct = 100 * fat_proj / FAT_QUOTA if FAT_QUOTA else 0.0
    _cf = C['R'] if _pct > 85 else (C['y'] if _pct > 65 else C['c'])
    _left = FAT_QUOTA - fat_proj
    _horiz = f" → budget lasts {_left/cores:.0f} h" if cores > 0 else ""
    A(f" {C['b']}{'Budget':<{LB}}{C['r']}campaign {C['c']}{fat_used}{C['r']} core-h · "
      f"project {_cf}{fat_proj}{C['r']}/{FAT_QUOTA} ({_cf}{_pct:.1f} %{C['r']}) · "
      f"left {C['g']}{_left}{C['r']} · burn {C['c']}{cores}{C['r']} core-h/h"
      f"{C['d']}{_horiz}{C['r']}")
    A(f" {C['b']}{'Method':<{LB}}{C['r']}GPAW LCAO/dzp · RPBE · h=0.16 Å · FermiDirac σ=0.1 "
      f"(E extrapolated σ→0) · BFGS fmax 0.05 eV/Å · 200 steps")
    A(f" {C['b']}{'Reference':<{LB}}{C['r']}{C['d']}E(H₂) = {E_H2:.6f} eV (h=0.16, d=0.7755 Å) · "
      f"ΔG_H = E_adslab − E_clean − ½E(H₂) + {ZPE} eV{C['r']}")
    A("")

    # ── active calculations ─────────────────────────────────────────────────
    if WIDE:
        CW = [18, 14, 11, 25, 18, 8, 8, 19, 12, 10, 5, 6]
        HD = ["ACTIVE RUNS", "CRYSTAL", "DEFECT", "GEOMETRY", "ADSORPTION",
              "SIDE", "ITER", "fmax → trend", "E [eV]", "~ΔG_H", "SCF", "NODE"]
    else:
        CW = [24, 8, 8, 19, 12, 10, 5, 6]
        HD = ["ACTIVE RUNS", "SIDE", "ITER", "fmax → trend", "E [eV]",
              "~ΔG_H", "SCF", "NODE"]
    A(" " + " ".join(f"{C['b']}{h:<{w}}{C['r']}" for h, w in zip(HD, CW)))
    A(f" {C['d']}{'─'*min(W-2, sum(CW)+len(CW))}{C['r']}")

    now = time.time()
    shown = 0
    for p in sorted(run, key=lambda x: x[3]):
        jid, arrf, arrk, jname, _, nc, elapsed, node = p[:8]
        cand = None
        # SLURM gives non-array jobs %a = 4294967294 (the "not an array"
        # sentinel), so the file is slurm_<jid>_4294967294.out. Only
        # slurm_<arrf>_<arrk>.out and slurm_<jid>.out were tried before, neither
        # of which exists, which is why the active-runs table reported that no
        # task had written a structure name.
        cands = [CDIR / f"slurm_{arrf}_{arrk}.out", CDIR / f"slurm_{jid}.out"]
        cands += sorted(CDIR.glob(f"slurm_{jid}_*.out"))
        for lp in cands:
            if lp.is_file():
                hits = re.findall(r"^>>> (\S+)", lp.read_text(errors="ignore"), re.M)
                if hits:
                    cand = hits[-1]
                    break
        if not cand:
            continue
        m, L = meta.get(cand, {}), live(cand)
        shown += 1
        ec = (L.get("clean") or {}).get("E")   # BFGS only = converged
        sides = [(k, v) for k, v in (("clean", L.get("clean")),
                                     ("adslab", L.get("adslab"))) if v] or [("clean", {})]
        for si, (side, st) in enumerate(sides):
            tr = st.get("trend") or []
            arrow = " ".join(f"{v:.3f}" for v in tr) if tr else "—"
            if len(tr) >= 2:
                mk, mc = (("↓", C['g']) if tr[-1] < tr[0] * 0.98
                          else ("↑", C['R']) if tr[-1] > tr[0] * 1.05 else ("→", C['y']))
            else:
                mk, mc = " ", ""
            fm = st.get("fmax")
            fc = (C['g'] if fm is not None and fm < 0.05 else
                  C['R'] if fm is not None and fm > 0.5 else C['y'] if fm is not None else "")
            E_conv = st.get("E")               # from the BFGS log = converged SCF
            E = E_conv if E_conv is not None else st.get("scf_E")
            unconv = E_conv is None and E is not None
            dgp = ""
            # ~ΔG_H only from converged energies on BOTH sides. A running SCF
            # iteration can be hundreds of eV off (measured: -853 eV against a
            # converged -440).
            if side == "adslab" and ec is not None and E_conv is not None:
                dgp = f"~{E_conv - ec - 0.5*E_H2 + ZPE:+.4f}"
            ph = side + (" ✓" if st.get("done") and fm is not None and fm < 0.05 else "")
            it = f"{st.get('step','—')}/200"
            first = si == 0
            zl = f"{m.get('formula','?')} {m.get('facet','?')}" + \
                 ("" if m.get("variant") == "base" else f" {m.get('variant','')}")
            row = " " + pad(zl if first else "", CW[0],
                            (C['b'] + C['m']) if first else "") + " "
            if WIDE:
                row += pad(f"{m.get('sg','?')} #{m.get('sgn','?')}" if first else "",
                           CW[1], C['d']) + " "
                row += pad(m.get('defect', '—') if first else "", CW[2], C['d']) + " "
                row += pad((f"{m.get('n_atoms','?')}at {m.get('a_A',0):.1f}×{m.get('b_A',0):.1f} "
                            f"hr{m.get('thickness_A',0):.1f} fix{m.get('n_fixed','?')}")
                           if first else "", CW[3], C['d']) + " "
                row += pad((f"{m.get('site','—')} {m.get('nn_sym','')} "
                            f"{m.get('d_HM_A',0):.2f}Å {m.get('coverage','')}")
                           if first else "", CW[4], C['d']) + " "
                k = 5
            else:
                k = 1
            row += pad(ph, CW[k], C['c']) + " "
            row += pad(it, CW[k+1], C['y']) + " "
            row += pad(arrow, CW[k+2] - 2, fc) + f" {mc}{mk}{C['r']} "
            # "≈" = energy from a RUNNING SCF iteration, not a converged one
            row += pad((("≈" if unconv else "") + f"{E:.4f}") if E is not None else "—",
                       CW[k+3], C['d'] if unconv else "") + " "
            row += pad(dgp, CW[k+4], (C['b'] + C['g']) if dgp else "") + " "
            row += pad(f"{(now-st.get('mtime',now))/60:.0f}m" if "mtime" in st else "—",
                       CW[k+5]) + " "
            row += pad(node if first else "", CW[k+6], C['d'])
            A(row)
        if not WIDE:
            A(f" {C['d']}   ↳ {m.get('sg','?')} #{m.get('sgn','?')} {m.get('lattice','?')} "
              f"{m.get('proto','?')} · {'sym' if m.get('symmetric') else 'asym'} slab, "
              f"term {m.get('term_top','?')}, {m.get('stoich','?')} · {m.get('defect','—')} · "
              f"{m.get('n_atoms','?')}at {m.get('a_A',0):.1f}×{m.get('b_A',0):.1f}Å "
              f"thk{m.get('thickness_A',0):.1f} fix{m.get('n_fixed','?')} "
              f"vac{m.get('vacuum_A',0):.0f} · H-{m.get('site','—')} on {m.get('nn_sym','')} "
              f"{m.get('d_HM_A',0):.2f}Å, {m.get('coverage','')} ML · "
              f"IBZ {(L.get('clean') or L.get('adslab') or {}).get('ibz','?')} k · "
              f"{jname} {jid} {nc}r {elapsed}{C['r']}")
    if shown == 0:
        A(f" {C['d']}no task has written a structure name into its log yet{C['r']}")
    A("")

    # ── ΔG_H: every structure with its run status ───────────────────────────
    # Not only the finished ones, so it is visible which structure lands next
    # and how far along it is. Progress is convergence distance (log fmax),
    # not elapsed time.
    SHOW_ALL = os.environ.get("FULL", "1") != "0"
    A(f" {C['b']}HYDROGEN ADSORPTION FREE ENERGY{C['r']}  {C['d']}ΔG_H < 0 exothermic · "
      f"|ΔG_H| < 0.1 eV = HER optimum (Nørskov) · ~ = provisional, relaxation unfinished · "
      f"progress = convergence distance log(fmax){C['r']}")
    RW = [16, 7, 11, 22, 10, 10, 10, 8, 8, 4]
    RH = ["COMPOUND", "FACET", "DEFECT", "RUN STATUS", "ΔG_H DFT", "@UMA geom",
          "ΔG_H UMA", "|err|", "STEPS", "HER"]
    A(" " + " ".join(f"{C['b']}{h:<{w}}{C['r']}" for h, w in zip(RH, RW)))
    A(f" {C['d']}{'─'*min(W-2, sum(RW)+len(RW))}{C['r']}")

    # collect the state of EVERY structure
    items = []
    for nm, r in rows.items():
        m = meta.get(nm, {})
        done = ok.get(nm)
        if done:
            try:
                dg = float(done["dGH_eV"]); u = float(done["gibbs_free_ml_eV"])
            except (ValueError, KeyError, TypeError) as _e:
                # A missing ΔG_H or UMA value must NOT drop the row. The
                # TypeError comes from float(None): DictReader yields None for
                # missing fields in SHORT rows (the runner writes 30 columns
                # while the header has 31).
                items.append(dict(o=4, nm=nm, m=m, status="✗ incomplete row", dg=None,
                                  s0=None, u=None, e=None, kr="—", prov=False))
                continue
            # step0 is OPTIONAL — deliberately not written on a restart from a trajectory
            s0 = None
            try:
                if done.get("dGH_step0_eV"):
                    s0 = float(done["dGH_step0_eV"])
            except ValueError:
                s0 = None
            items.append(dict(o=0, nm=nm, m=m, status="✓ done", dg=dg, s0=s0, u=u,
                              e=abs(dg - u), kr=f"{done.get('steps_clean','?')}/"
                              f"{done.get('steps_adslab','?')}", prov=False))
            continue
        if nm in err:
            items.append(dict(o=4, nm=nm, m=m, status=f"✗ {err[nm]['status'][:16]}",
                              dg=None, s0=None,
                              u=float(r["gibbs_free_ml_eV"]), e=None, kr="—", prov=False))
            continue
        L = live(nm)
        cl, ad = L.get("clean"), L.get("adslab")
        if not cl and not ad:
            items.append(dict(o=3, nm=nm, m=m, status="pending", dg=None, s0=None,
                              u=float(r["gibbs_free_ml_eV"]), e=None, kr="—", prov=False))
            continue
        side, st = ("adslab", ad) if ad else ("clean", cl)
        pct = conv_pct(st.get("hist"))
        if pct is None:
            # No BFGS step written yet = the FIRST single-point on the input
            # geometry is running. There is no convergence percentage, but the
            # SCF iteration exists and is the only progress signal available.
            status = f"{side:<6} 1st SCF, iter {st.get('scf','—')}"
        else:
            status = f"{side:<6} {bar(pct)} {pct:>3}%"
        dg = None
        # ONLY "E" (from the BFGS log = converged SCF), never "scf_E" (a running iteration).
        if ad and cl and ad.get("E") is not None and cl.get("E") is not None:
            dg = ad["E"] - cl["E"] - 0.5 * E_H2 + ZPE
        u = float(r["gibbs_free_ml_eV"])
        items.append(dict(o=1 if side == "adslab" else 2, nm=nm, m=m, status=status,
                          dg=dg, s0=None, u=u,
                          e=abs(dg - u) if dg is not None else None,
                          kr=f"{(cl or {}).get('step','—')}/{(ad or {}).get('step','—')}",
                          prov=True))

    # INVARIANT: the row count MUST equal the structure count. Without this
    # check a silently dropped row can only be caught by counting by hand — and
    # that has already happened twice in this project.
    missing = sorted(set(rows) - {d["nm"] for d in items})
    if missing:
        for nm in missing:
            r_ = resmap_dbg.get(nm, {})
            items.append(dict(o=4, nm=nm, m=meta.get(nm, {}),
                              status=f"✗ DROPPED ({r_.get('status', 'not in CSV')})",
                              dg=None, s0=None, u=None, e=None, kr="—", prov=False))
    items.sort(key=lambda d: (d["o"], d["nm"]))
    if not SHOW_ALL:
        items = [d for d in items if d["o"] <= 2]
    for d in items:
        m = d["m"]
        her = d["dg"] is not None and not d["prov"] and abs(d["dg"]) < 0.1
        pre = "~" if d["prov"] and d["dg"] is not None else ""
        col = (C['g'] if d["o"] == 0 else C['y'] if d["o"] == 1
               else C['c'] if d["o"] == 2 else C['R'] if d["o"] == 4 else C['d'])
        ecol = "" if d["e"] is None else (C['g'] if d["e"] < 0.1
                                         else C['y'] if d["e"] < 0.3 else C['R'])
        A(" " + pad(m.get("formula", d["nm"]), RW[0], C['b'] if her else "")
          + " " + pad(m.get("facet", "?"), RW[1])
          + " " + pad(m.get("defect", "—"), RW[2], C['d'])
          + " " + pad(d["status"], RW[3], col)
          + " " + pad(f"{pre}{d['dg']:+.4f}" if d["dg"] is not None else "—", RW[4],
                      C['b'] if not d["prov"] else C['d'])
          + " " + pad(f"{d['s0']:+.4f}" if d["s0"] is not None else "—", RW[5], C['d'])
          + " " + pad(f"{d['u']:+.4f}" if d["u"] is not None else "—", RW[6])
          + " " + pad(f"{pre}{d['e']:.4f}" if d["e"] is not None else "—", RW[7], ecol)
          + " " + pad(d["kr"], RW[8])
          + " " + pad("★" if her else "", RW[9], C['g']))

    fin = [d for d in items if d["o"] == 0]
    if fin:
        es = sorted(x["e"] for x in fin)
        near = sum(1 for x in fin if abs(x["dg"]) < 0.1)
        A(f" {C['d']}{'─'*min(W-2, sum(RW)+len(RW))}{C['r']}")
        A(f" {C['b']}MAE {sum(es)/len(es):.4f} eV{C['r']}  median {es[len(es)//2]:.4f}  "
          f"max {es[-1]:.4f}  n = {len(es)}   ·   "
          f"{C['b']}{C['g']}HER ★ |ΔG_H|<0.1 eV: {near}{C['r']}   ·   "
          f"{C['d']}states: {sum(1 for d in items if d['o']==0)} done, "
          f"{sum(1 for d in items if d['o']==1)} in adslab, "
          f"{sum(1 for d in items if d['o']==2)} in clean, "
          f"{sum(1 for d in items if d['o']==3)} pending, "
          f"{sum(1 for d in items if d['o']==4)} failed{C['r']}"
          + (f"   {C['R']}{C['b']}⚠ INVARIANT: {len(items)} rows ≠ {len(rows)} "
             f"structures{C['r']}" if len(items) != len(rows) else
             f"   {C['g']}✓ {len(items)}/{len(rows)}{C['r']}"))

    flg = CDIR / "campaign_flags.txt"
    if flg.is_file():
        bad = [l for l in flg.read_text().splitlines() if l.startswith("-")]
        if bad:
            A(f"\n {C['R']}{C['b']}⚠ WATCHDOG: {len(bad)} issues{C['r']}")
            for b in bad[:5]:
                A(f"   {C['R']}{cut(b, W-6)}{C['r']}")
    A(f"\n {C['d']}Ctrl+C stops the monitor (jobs keep running) · watchdog: "
      f"campaign_status.txt / campaign_flags.txt · "
      f"{'wide' if WIDE else 'narrow'} mode, {W} cols (override with COLS=){C['r']}")
    print("\n".join(out))


if __name__ == "__main__":
    once = "--once" in sys.argv
    ref = int(os.environ.get("REFRESH", "30"))
    while True:
        try:
            render()
        except Exception:
            import traceback
            traceback.print_exc()
        if once:
            break
        time.sleep(ref)
