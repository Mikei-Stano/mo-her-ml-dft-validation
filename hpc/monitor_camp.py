#!/usr/bin/env python3
"""monitor_camp.py — produkčný monitor DFT kampane (ΔG_H validácia UMA).

Štvrtý do rodiny:
    monitor_uma.sh   →  UMA/AdsorbML na GPU
    monitor_val.sh   →  konvergenčné validácie (k-mriežka, h, σ)
    monitor_dft.sh   →  starý DFT beh (gpaw_calculations)
    monitor_camp.py  →  TÁTO kampaň (campaign_dft) + kryštalografia

Prečo Python a nie bash+awk ako ostatné: terminácia povrchu, supercela, adsorpčné
miesto a pokrytie sa nedajú odvodiť z názvu súboru — počítajú sa z polôh atómov,
bunky a FixAtoms cez ASE. Kryštalografia sa cachuje do campaign_meta.json.

ADAPTÍVNA ŠÍRKA: pri termináli ≥ 172 znakov ide kryštalografia do samostatných
stĺpcov, pri užšom na potichý podriadok pod každou štruktúrou. Hlavička má pevnú
menovku (11 znakov), takže hodnoty sedia presne pod sebou.

PRIEBEŽNÁ ΔG_H: hneď ako je rozbehnutá clean aj adslab strana, počíta sa
~ΔG_H = E_adslab − E_clean − ½E(H₂) + ZPE z posledných BFGS energií. Je to
predbežné číslo (ani jedna strana nemusí byť dokončená), preto vlnovka.
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

# Prostredie klastra — prepíš cez env, ak beží pod iným účtom či projektom.
SLURM_USER    = os.environ.get("SLURM_USER", os.environ.get("USER", ""))
SLURM_ACCOUNT = os.environ.get("SLURM_ACCOUNT", "")
SINCE         = os.environ.get("SACCT_SINCE", "2026-07-29")
SINCE_PROJECT = os.environ.get("SACCT_SINCE_PROJECT", "2026-01-01")
META_VER = 2          # zmena formátu polí → vynúti prepočet cache
E_H2 = -6.466266559623192          # h=0.16, RPBE/dzp, zrelaxovaná d(H–H) = 0.7755 Å
ZPE = 0.24
# Alokácia je na QOS p2061-26-2_fat: GrpTRESMins billing=48 000 000 min / 60
# = 800 000 core·h. Overené: `sacctmgr show qos p2061-26-2_fat`. Generická CPU
# QOS p2061-26-2 má billing=0, takže cpu_long/cpu_short sú pre projekt zavreté
# a celý rozpočet kampane je v týchto 800 000 na FAT uzloch.
FAT_QUOTA = int(os.environ.get("FAT_QUOTA", "800000"))   # core·h, alokácia na FAT
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

# Priestorové grupy z opravených bulk builderov (scripts/generate_structures.py,
# každý s citáciou, 2026-07-28).
PHASE = {
    "MoP":     dict(f="MoP",     proto="WC-typ (Bₕ)",         sg="P-6m2",   n=187, sys="hex", anion="P"),
    "MoS2":    dict(f="MoS₂",    proto="2H polytyp",          sg="P6₃/mmc", n=194, sys="hex", anion="S"),
    "MoSe2":   dict(f="MoSe₂",   proto="2H polytyp",          sg="P6₃/mmc", n=194, sys="hex", anion="Se"),
    "Mo2C":    dict(f="Mo₂C",    proto="ortoromb. ζ-Fe₂N",    sg="Pbcn",    n=60,  sys="ort", anion="C"),
    "Mo2N":    dict(f="γ-Mo₂N",  proto="defekt. kam. soľ",    sg="Fm-3m",   n=225, sys="kub", anion="N"),
    "MoB":     dict(f="MoB",     proto="CrB-typ",             sg="Cmcm",    n=63,  sys="ort", anion="B"),
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
    """Zarovnanie na presnú šírku. Farba sa pripočíta AŽ po výpočte dĺžky,
    inak by ANSI kódy rozhodili stĺpce."""
    x = cut(x, n)
    return (f"{col}{x}{C['r']}" if col else x) + " " * (n - len(x))



def conv_pct(fm, target=0.05):
    """Pokrok relaxácie v %, meraný na LOGARITMICKEJ škále fmax.

    Počet zostávajúcich krokov sa dopredu vedieť nedá, ale vzdialenosť
    k cieľu áno: fmax klesá ~exponenciálne, takže log(f0/f)/log(f0/target)
    je monotónny odhad toho, "koľko z cesty je za nami". 0 % = na štarte,
    100 % = fmax <= target. Nie je to čas, je to konvergenčná vzdialenosť.
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


# ── kryštalografia zo skutočnej geometrie ────────────────────────────────────
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
    # terminácia = druhy do 1 Å od najvyššieho / najnižšieho atómu
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
    d["stoich"] = ("stech." if var == "base"
                   else f"red. ({an}-def.)" if var.startswith("vac") else "dopovaný")
    d.update(ads="—", site="—", coverage="—", h_height_A=None)
    if af.is_file():
        ads = read(str(af))
        if len(ads) == len(slab) + 1 and ads.get_chemical_symbols()[-1] == "H":
            hp, sub = ads.positions[-1], ads.positions[:-1]
            dist = np.linalg.norm(sub - hp, axis=1)
            dmin = float(dist.min())
            near = int((dist < dmin * 1.25).sum())   # koordinácia H = typ miesta
            d.update(ads="H chemisorpcia", nn=near, d_HM_A=dmin,
                     nn_sym=ads[int(np.argmin(dist))].symbol,
                     h_height_A=float(hp[2] - z.max()),
                     site={1: "top", 2: "brdg", 3: "holl3", 4: "holl4"}.get(near, f"{near}f"))
            if d["n_top"]:
                d["coverage"] = f"1/{d['n_top']}"
    return d


def build_meta():
    rows = {r["slab_name"]: r for r in csv.DictReader(open(RANKED))}
    out = {}
    if META.is_file():
        try:
            out = json.loads(META.read_text())
        except Exception:
            out = {}
    ch = False
    for nm, r in rows.items():
        # Kľúč cache musí zahŕňať aj KANDIDÁTA — po väzbovom prerankovaní sa
        # candidate_file zmenil a monitor inak ukazoval starú (nenaväzanú)
        # adsorpčnú geometriu, teda tvrdil, že oprava neplatí.
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


# ── živý stav jednej štruktúry ───────────────────────────────────────────────
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
            # POČET KROKOV = počet zapísaných riadkov, NIE index z posledného.
            # Po restarte ASE začne číslovať od 0 a pripíše nové riadky do toho
            # istého logu, takže index posledného riadku hlásil "0/200" aj pri
            # 18 už odpracovaných krokoch (MoS2_(110)_vacS po oprave kandidáta).
            s["step"] = len(fm)
            s["restarts"] = max(0, lg.read_text(errors="ignore").count("Step     Time") - 1)
            s["fmax"] = fm[-1] if fm else None
            s["trend"] = fm[-3:]
            s["hist"] = fm            # celá história → pokrok konvergencie
        t = CDIR / f"{name}_{side}.txt"
        if t.is_file():
            s["mtime"] = t.stat().st_mtime
            tail = t.read_text(errors="ignore")[-6000:]
            # ([^|]+) namiesto (\S+): GPAW pripája konvergenčnú značku 'c' priamo
            # k číslu bez medzery ("-783.723917c|"), takže \S+ zhltlo aj rúrku
            # a float() padol → E sa v monitore ukazovalo ako "—".
            it = re.findall(r"^\|iter:\s*(\d+)\|[^|]*\|([^|]+)", tail, re.M)
            if it:
                s["scf"] = int(it[-1][0])
                # Energiu ber z POSLEDNÉHO riadku, ktorý sa dá naparsovať. GPAW
                # na začiatku SCF cyklu vypíše riadok s PRÁZDNYM energetickým
                # polom ("|iter: 28| 13:24:30 |          |"), takže brať slepo
                # it[-1] znamenalo E = "—" hoci SCF dávno beží (γ-Mo₂N, MoB).
                for _, raw in reversed(it):
                    try:
                        s["scf_E"] = float(raw.strip().rstrip("c").strip())
                        break
                    except ValueError:
                        continue
            s["done"] = "Free energy" in tail
            try:    # IBZ je v HLAVIČKE logu, nie v chvoste
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
        # "camp*" = array joby pôvodnej kampane, "c-<slab>" = per-štruktúrové joby
        # po prechode na konfiguráciu B, "np-" = škálovacie testy. Bez "c-" hlásil
        # monitor 1 job / 144 jadier, kým reálne bežalo 32 jobov na 4176 jadrách.
        # "s0-" a "strain-" = doplnkové single-pointy po dokončení kampane; tie
        # počítajú UŽ HOTOVÉ štruktúry (inú veličinu), takže sa môžu objaviť aj
        # v aktívnych výpočtoch aj ako "✓ hotové" v hlavnej tabuľke.
        if len(p) < 8 or not p[3].startswith(("camp", "c-", "np-", "s0-", "strain")):
            continue
        if p[4] == "RUNNING":
            run.append(p)
            cores += int(p[5] or 0)
            nodes.update(x for x in p[7].replace("[", " ").replace("]", " ")
                         .replace(",", " ").split() if x.startswith("cn"))
            arrays.add(p[1])
            # SLURM elapsed má formát [D-]HH:MM:SS. Dni sú DNI, nie ďalšia
            # 60-ková jednotka — pôvodný kód robil `sec = sec*60 + v` aj na poli
            # dní, takže "1-01:56:15" (25.94 h) vyšlo ako 61.94 h. Overené:
            # buggy 61.9375 h vs správne 25.9375 h.
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
    # FAT sa účtuje proti QOS p2061-26-2_fat (48 000 000 billing-min = 800 000
    # core·h), NIE proti účtu — assoc má GrpTRESMins prázdne. Generická CPU QOS
    # p2061-26-2 má alokáciu 0, preto sú cpu_long/cpu_short nedostupné a celý
    # rozpočet je v týchto 800 000. Dopyt na celú históriu trvá 0.03 s.
    fat = sh(f"sacct -u {SLURM_USER} -S {SINCE} -X -n -o NCPUS,ElapsedRaw,AllocTRES%40 "
             "--parsable2 | awk -F'|' '$3 !~ /gres.gpu/ {t+=$1*$2} END{printf \"%.0f\", t/3600}'")
    fat_used = int(fat) if (fat or "").strip().isdigit() else 0
    fp = sh(f"sacct -A {SLURM_ACCOUNT} -S {SINCE_PROJECT} -X -n -P -o NCPUS,ElapsedRaw,Partition "
            "| awk -F'|' '$3 ~ /hm/ {t+=$1*$2} END{printf \"%.0f\", t/3600}'")
    fat_proj = int(fp) if (fp or "").strip().isdigit() else 0
    # ETA NESMIE byť zostatok/rate — to predpokladá sekvenčný beh, kým 63 zo 74
    # zostávajúcich beží SÚBEŽNE a dobehnú približne spolu. Správny model je
    # vlnový: koľko vĺn po `concurrency` štruktúrach ešte treba, × priemerný
    # čas na štruktúru (z už dokončených, stĺpec wall_s).
    walls = []
    for r_ in ok.values():
        try:
            walls.append(float(r_["wall_s"]) / 3600)
        except (KeyError, ValueError, TypeError):
            pass
    mean_wall = sum(walls) / len(walls) if walls else None
    remain = TOTAL - len(ok)
    conc = max(1, len(run))
    # rate NESMIE byť hotové/vek_najstaršieho_jobu — po restarte kampane je vek
    # jobov malý, kým hotových je veľa, a vyšlo "36.7/h, ETA 205 h" (nezmysel).
    # Priepustnosť je conc/mean_wall: `conc` štruktúr súbežne, každá mean_wall h.
    rate = (conc / mean_wall) if mean_wall else 0.0
    if mean_wall and remain > 0:
        waves = -(-remain // conc)                     # ceil
        eta = waves * mean_wall
    else:
        eta = None

    out = ["\033[H\033[2J"]
    A = out.append
    A(f"{C['b']}╔{'═'*(W-2)}╗{C['r']}")
    t1 = "DFT KAMPAŇ ⇄ PERUN   ΔG_H validácia AdsorbML/UMA"
    t2 = f"{len(run)} jobov · {cores} jadier · {time.strftime('%H:%M:%S')}"
    A(f"{C['b']}║{C['r']} {t1:<{max(1, W-5-len(t2))}}{C['d']}{t2}{C['r']} {C['b']}║{C['r']}")
    A(f"{C['b']}╚{'═'*(W-2)}╝{C['r']}")

    LB = 11        # pevná menovka → všetky hodnoty začínajú v rovnakom stĺpci
    bn = 24
    fl = int(len(ok) / TOTAL * bn)
    # "od štartu behu" bolo zavádzajúce: `oldest` je vek NAJSTARŠIEHO bežiaceho
    # jobu, čo sa po každom restarte kampane vynuluje. Pomenované poctivo.
    _ids = sorted(arrays, key=lambda s: (len(s), s))
    if len(_ids) > 6:                    # per-štruktúrových jobov je 27+, nevojdú sa
        _idtxt = f"{len(_ids)}× ({_ids[0]}…{_ids[-1]})"
    else:
        _idtxt = " ".join(_ids) or "—"
    A(f" {C['d']}{'joby':<{LB}}{_idtxt:<44}"
      f"najstarší job beží {oldest:.2f} h{C['r']}")
    A("")
    A(f" {C['b']}{'Progres':<{LB}}{C['r']}[{C['g']}{'█'*fl}{C['r']}{'░'*(bn-fl)}] "
      f"{C['b']}{len(ok)}/{TOTAL}{C['r']} ({100*len(ok)/TOTAL:.0f} %)"
      f"    rate {rate:.1f}/h    ETA {f'{eta:.1f} h' if eta else '—'}"
      f"   {C['d']}({f'{mean_wall:.2f} h/štr.' if mean_wall else '—'}, "
      f"{-(-remain // conc) if mean_wall else '?'} vĺn × {conc} súbežne){C['r']}")
    A(f" {C['b']}{'Fronta':<{LB}}{C['r']}beží {C['g']}{len(run)}{C['r']} · čaká {pend} · "
      f"hotové {C['g']}{len(ok)}{C['r']} · zlyhalo "
      f"{C['R'] if err else ''}{len(err)}{C['r']} · zostáva {TOTAL-len(ok)}")
    A(f" {C['b']}{'Zdroje':<{LB}}{C['r']}{C['c']}{cores}{C['r']} FAT jadier z {PART_CORES} "
      f"({100*cores/PART_CORES:.0f} %) · {C['c']}{len(nodes)}{C['r']}/{NODES_TOT} uzlov · "
      f"{cores} vlákien (OMP=1, ThreadsPerCore=1) · 3500 MB/jadro")
    _pct = 100 * fat_proj / FAT_QUOTA if FAT_QUOTA else 0.0
    _cf = C['R'] if _pct > 85 else (C['y'] if _pct > 65 else C['c'])
    _left = FAT_QUOTA - fat_proj
    _horiz = f" → zostatok na {_left/cores:.0f} h" if cores > 0 else ""
    A(f" {C['b']}{'FAT':<{LB}}{C['r']}kampaň {C['c']}{fat_used}{C['r']} core·h · "
      f"projekt {_cf}{fat_proj}{C['r']}/{FAT_QUOTA} ({_cf}{_pct:.1f} %{C['r']}) · "
      f"zostáva {C['g']}{_left}{C['r']} · tempo {C['c']}{cores}{C['r']} core·h/h"
      f"{C['d']}{_horiz}{C['r']}")
    A(f" {C['b']}{'Metóda':<{LB}}{C['r']}GPAW LCAO/dzp · RPBE · h=0.16 Å · FermiDirac σ=0.1 "
      f"(E extrapolovaná σ→0) · BFGS fmax 0.05 eV/Å · 200 krokov")
    A(f" {C['b']}{'Referencia':<{LB}}{C['r']}{C['d']}E(H₂) = {E_H2:.6f} eV (h=0.16, d=0.7755 Å) · "
      f"ΔG_H = E_adslab − E_clean − ½E(H₂) + {ZPE} eV{C['r']}")
    A("")

    # ── aktívne výpočty ──────────────────────────────────────────────────────
    if WIDE:
        CW = [18, 14, 11, 25, 18, 8, 8, 19, 12, 10, 5, 6]
        HD = ["AKTÍVNE VÝPOČTY", "KRYŠTÁL", "DEFEKT", "GEOMETRIA", "ADSORPCIA",
              "FÁZA", "ITER", "fmax → trend", "E [eV]", "~ΔG_H", "SCF", "UZOL"]
    else:
        CW = [24, 8, 8, 19, 12, 10, 5, 6]
        HD = ["AKTÍVNE VÝPOČTY", "FÁZA", "ITER", "fmax → trend", "E [eV]",
              "~ΔG_H", "SCF", "UZOL"]
    A(" " + " ".join(f"{C['b']}{h:<{w}}{C['r']}" for h, w in zip(HD, CW)))
    A(f" {C['d']}{'─'*min(W-2, sum(CW)+len(CW))}{C['r']}")

    now = time.time()
    shown = 0
    for p in sorted(run, key=lambda x: x[3]):
        jid, arrf, arrk, jname, _, nc, elapsed, node = p[:8]
        cand = None
        # Non-array joby dostanú od SLURMu %a = 4294967294 (sentinel "nie je array"),
        # takže súbor je slurm_<jid>_4294967294.out. Predtým sa skúšalo len
        # slurm_<arrf>_<arrk>.out a slurm_<jid>.out — ani jedno neexistuje, preto
        # tabuľka aktívnych výpočtov hlásila "žiadny task nezapísal názov".
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
        ec = (L.get("clean") or {}).get("E")   # len BFGS = konvergované
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
            E_conv = st.get("E")               # z BFGS logu = konvergované SCF
            E = E_conv if E_conv is not None else st.get("scf_E")
            unconv = E_conv is None and E is not None
            dgp = ""
            # ~ΔG_H iba z konvergovaných energií OBOCH strán. Bežiaca SCF iterácia
            # môže byť o stovky eV mimo (namerané: -853 eV pri konvergovanej -440).
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
            # "≈" = energia z BEŽIACEJ SCF iterácie, nie konvergovaná
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
              f"hr{m.get('thickness_A',0):.1f} fix{m.get('n_fixed','?')} "
              f"vák{m.get('vacuum_A',0):.0f} · H-{m.get('site','—')} na {m.get('nn_sym','')} "
              f"{m.get('d_HM_A',0):.2f}Å, {m.get('coverage','')} ML · "
              f"IBZ {(L.get('clean') or L.get('adslab') or {}).get('ibz','?')} k · "
              f"{jname} {jid} {nc}r {elapsed}{C['r']}")
    if shown == 0:
        A(f" {C['d']}žiadny task nezapísal názov štruktúry do logu{C['r']}")
    A("")

    # ── ΔG_H: VŠETKÝCH 83 so stavom výpočtu ─────────────────────────────────
    # Nielen dokončené — aby bolo vidno, ktorá štruktúra pribudne ako ďalšia
    # a ako daleko je. Pokrok je konvergenčná vzdialenosť (log fmax), nie čas.
    SHOW_ALL = os.environ.get("FULL", "1") != "0"
    A(f" {C['b']}GIBBSOVA ADSORPČNÁ ENERGIA{C['r']}  {C['d']}ΔG_H < 0 exotermická · "
      f"|ΔG_H| < 0.1 eV = HER optimum (Nørskov) · ~ = priebežná, relaxácia nedokončená · "
      f"pokrok = konvergenčná vzdialenosť log(fmax){C['r']}")
    RW = [16, 7, 11, 22, 10, 10, 10, 8, 8, 4]
    RH = ["ZLÚČENINA", "ROVINA", "DEFEKT", "STAV VÝPOČTU", "ΔG_H DFT", "na UMA g.",
          "ΔG_H UMA", "|err|", "KROKOV", "HER"]
    A(" " + " ".join(f"{C['b']}{h:<{w}}{C['r']}" for h, w in zip(RH, RW)))
    A(f" {C['d']}{'─'*min(W-2, sum(RW)+len(RW))}{C['r']}")

    # zozbieraj stav pre KAŽDÚ štruktúru
    items = []
    for nm, r in rows.items():
        m = meta.get(nm, {})
        done = ok.get(nm)
        if done:
            try:
                dg = float(done["dGH_eV"]); u = float(done["gibbs_free_ml_eV"])
            except (ValueError, KeyError, TypeError) as _e:
                # ΔG_H alebo UMA hodnota chýba/je None → riadok sa NESMIE zahodiť.
                # TypeError vzniká z float(None): DictReader dá None pre chýbajúce
                # polia v KRÁTKYCH riadkoch (runner píše 30 stĺpcov, hlavička má 31).
                items.append(dict(o=4, nm=nm, m=m, stav="✗ neúplný riadok", dg=None,
                                  s0=None, u=None, e=None, kr="—", prov=False))
                continue
            # step0 je VOLITEĽNÝ — pri restarte z trajektórie sa zámerne nezapisuje
            s0 = None
            try:
                if done.get("dGH_step0_eV"):
                    s0 = float(done["dGH_step0_eV"])
            except ValueError:
                s0 = None
            items.append(dict(o=0, nm=nm, m=m, stav=f"✓ hotové", dg=dg, s0=s0, u=u,
                              e=abs(dg - u), kr=f"{done.get('steps_clean','?')}/"
                              f"{done.get('steps_adslab','?')}", prov=False))
            continue
        if nm in err:
            items.append(dict(o=4, nm=nm, m=m, stav=f"✗ {err[nm]['status'][:16]}",
                              dg=None, s0=None,
                              u=float(r["gibbs_free_ml_eV"]), e=None, kr="—", prov=False))
            continue
        L = live(nm)
        cl, ad = L.get("clean"), L.get("adslab")
        if not cl and not ad:
            items.append(dict(o=3, nm=nm, m=m, stav="čaká", dg=None, s0=None,
                              u=float(r["gibbs_free_ml_eV"]), e=None, kr="—", prov=False))
            continue
        side, st = ("adslab", ad) if ad else ("clean", cl)
        pct = conv_pct(st.get("hist"))
        if pct is None:
            # Ešte nezapísala BFGS krok = beží PRVÝ single-point na vstupnej
            # geometrii. Percento konvergencie neexistuje, ale SCF iterácia áno
            # a je to jediné, čo o pokroku vypovedá.
            stav = f"{side:<6} 1. SCF, iter {st.get('scf','—')}"
        else:
            stav = f"{side:<6} {bar(pct)} {pct:>3}%"
        dg = None
        # LEN "E" (z BFGS logu = konvergované SCF), nikdy "scf_E" (bežiaca iterácia).
        if ad and cl and ad.get("E") is not None and cl.get("E") is not None:
            dg = ad["E"] - cl["E"] - 0.5 * E_H2 + ZPE
        u = float(r["gibbs_free_ml_eV"])
        items.append(dict(o=1 if side == "adslab" else 2, nm=nm, m=m, stav=stav,
                          dg=dg, s0=None, u=u,
                          e=abs(dg - u) if dg is not None else None,
                          kr=f"{(cl or {}).get('step','—')}/{(ad or {}).get('step','—')}",
                          prov=True))

    # INVARIANT: počet riadkov tabuľky sa MUSÍ rovnať počtu štruktúr. Bez tejto
    # kontroly sa tichý výpadok riadku dá odhaliť len ručným prepočítaním — a to
    # sa v tomto projekte už dvakrát stalo (raz som ho aj nesprávne odvolal).
    missing = sorted(set(rows) - {d["nm"] for d in items})
    if missing:
        for nm in missing:
            r_ = resmap_dbg.get(nm, {})
            items.append(dict(o=4, nm=nm, m=meta.get(nm, {}),
                              stav=f"✗ VYPADOL ({r_.get('status', 'nie je v CSV')})",
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
          + " " + pad(d["stav"], RW[3], col)
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
        A(f" {C['b']}MAE {sum(es)/len(es):.4f} eV{C['r']}  medián {es[len(es)//2]:.4f}  "
          f"max {es[-1]:.4f}  n = {len(es)}   ·   "
          f"{C['b']}{C['g']}HER ★ |ΔG_H|<0.1 eV: {near}{C['r']}   ·   "
          f"{C['d']}stavy: {sum(1 for d in items if d['o']==0)} hotové, "
          f"{sum(1 for d in items if d['o']==1)} v adslabe, "
          f"{sum(1 for d in items if d['o']==2)} v cleane, "
          f"{sum(1 for d in items if d['o']==3)} čaká, "
          f"{sum(1 for d in items if d['o']==4)} zlyhalo{C['r']}"
          + (f"   {C['R']}{C['b']}⚠ INVARIANT: {len(items)} riadkov ≠ {len(rows)} "
             f"štruktúr{C['r']}" if len(items) != len(rows) else
             f"   {C['g']}✓ {len(items)}/{len(rows)}{C['r']}"))

    flg = CDIR / "campaign_flags.txt"
    if flg.is_file():
        bad = [l for l in flg.read_text().splitlines() if l.startswith("-")]
        if bad:
            A(f"\n {C['R']}{C['b']}⚠ WATCHDOG: {len(bad)} problémov{C['r']}")
            for b in bad[:5]:
                A(f"   {C['R']}{cut(b, W-6)}{C['r']}")
    A(f"\n {C['d']}Ctrl+C ukončí monitor (joby bežia ďalej) · watchdog: "
      f"campaign_status.txt / campaign_flags.txt · "
      f"{'široký' if WIDE else 'úzky'} režim, {W} zn. (COLS= prepíše){C['r']}")
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
