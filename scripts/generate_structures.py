"""
Generate POSCAR files for Mo compounds
Creates surface slabs using literature lattice parameters and ASE
"""

import os
from pathlib import Path
from ase import Atoms
from ase.io import write
from ase.constraints import FixAtoms
from ase.build import fcc111, fcc100, surface
import numpy as np


def create_mos2_bulk():
    """2H-MoS2 (molybdenit, C7-typ), P6_3/mmc (#194). OPRAVENÉ 2026-07-28.

    Pôvodná hand-coded verzia mala DVE nezávislé chyby:
      (a) len 3 atómy (Z=1) v bunke s c=12.295 Å → vdW medzera 9.34 Å namiesto
          ~2.98 Å, teda elektronicky odviazaná monovrstva, nie bulk.
          Hustota 2.4999 vs exp. 4.997 g/cm³ = PRESNE polovica.
      (b) S na dvoch rôznych podmriežkach (1/3,2/3) a (2/3,1/3) → OKTAEDRICKÁ
          koordinácia Mo, symetria P-3m1 (#164) = metalický 1T-MoS2, nie
          polovodivý 2H-MoS2 s trigonálne prizmatickou koordináciou.

    Zdroj: COD 9007660 / AMCSD 0009788 = B. Schönfeld, J. J. Huang, S. C. Moss,
    Acta Cryst. B 39, 404 (1983), DOI 10.1107/S0108768183002645.
    Wyckoff: Mo 2c (1/3, 2/3, 1/4); S 4f (1/3, 2/3, z), z = 0.621(4)
    (z podľa COD 1010993, Dickinson & Pauling, JACS 1923).

    Overené po oprave: 6 atómov (Mo2S4), min. Mo–S 2.418 Å (exp. 2.41 Å,
    MP mp-2815: 2.42 Å), vdW medzera 2.975 Å, hustota 4.997 g/cm³
    (exp. 4.997 g/cm³, 0.0 % chyba), spglib potvrdzuje P6_3/mmc.
    """
    from ase.spacegroup import crystal
    return crystal(
        symbols=["Mo", "S"],
        basis=[(1 / 3, 2 / 3, 0.25),      # Mo, Wyckoff 2c
               (1 / 3, 2 / 3, 0.621)],    # S,  Wyckoff 4f
        spacegroup=194,                   # P6_3/mmc (2H-MoS2, C7-typ)
        cellpar=[3.161, 3.161, 12.295, 90, 90, 120],
    )


def create_mose2_bulk():
    """2H-MoSe2 (molybdenit-typ), P6_3/mmc (#194). OPRAVENÉ 2026-07-28.

    Tie isté dve chyby ako MoS2: (1) Se nad Mo na (1/3,2/3) a pod Mo na
    (2/3,1/3) → trojuholníky pretočené o 60° → OKTAEDRICKÁ (1T) koordinácia
    namiesto trigonálne prizmatickej (2H); 1T a 2H sú elektronicky odlišné fázy.
    (2) iba 3 atómy (Z=1) pri c=12.995 Å → izolovaná monovrstva + 9.88 Å vákua,
    nie bulk. Hustota 3.4631 vs exp. 6.90 g/cm³.

    Zdroj: COD 2310945 = P. B. James, M. T. Lavik, "The crystal structure of
    MoSe2", Acta Cryst. 16 (1963) 1183, DOI 10.1107/S0365110X6300311X.
    Wyckoff: Mo 2c (1/3, 2/3, 1/4); Se 4f (1/3, 2/3, 0.625); a=3.288, c=12.900.

    Overené po oprave: 6 atómov (Mo2Se4), min. Mo–Se 2.4907 Å (trigonálna
    prizma, 6× v zákryte), medzivrstvové Se–Se 3.7422 Å (vdW),
    hustota 6.9814 g/cm³ vs exp. 6.90 g/cm³ (+1.2 %).
    """
    from ase.spacegroup import crystal
    return crystal(
        symbols=["Mo", "Se"],
        basis=[(1 / 3, 2 / 3, 0.25),      # Mo, Wyckoff 2c
               (1 / 3, 2 / 3, 0.625)],    # Se, Wyckoff 4f
        spacegroup=194,                   # P6_3/mmc (2H-MoS2 typ)
        cellpar=[3.288, 3.288, 12.900, 90, 90, 120],
    )


def create_mop_bulk():
    """MoP, WC-typ (B_h), hexagonálny P-6m2 (#187). OPRAVENÉ 2026-07-28.

    Pôvodná hand-coded verzia stavala MoP ako KUBICKÝ CsCl-typ (B2), a=3.240 Å,
    Mo (0,0,0) / P (½,½,½). Taká fáza NEEXISTUJE. Nešlo o „mierne zlú mriežku":
    iná Bravaisova mriežka, iná koordinačná geometria (8-násobná kubická vs
    6-násobná trigonálno-prizmatická), Mo–P 2.806 vs 2.45 Å (+14.6 %),
    objem +18.7 %, hustota 6.197 vs exp. 7.34 g/cm³ (−15.8 %).
    Znehodnotilo VŠETKY MoP DFT aj ML výsledky.

    Zdroj mriežkových parametrov: Kumar, N. et al., "Extremely high conductivity
    observed in the triple point topological metal MoP", Nat. Commun. 10, 2475
    (2019), DOI 10.1038/s41467-019-10126-y — doslovne: "WC-type hexagonal
    crystal structure ... space group P-6m2 (No. 187) ... a = b = 3.22 Å and
    c = 3.19 Å".
    Zdroj Wyckoff pozícií: AFLOW prototyp WC (B_h) = AB_hP2_187_a_d-001,
    Mehl, M. J. et al., Comput. Mater. Sci. 136, S1 (2017),
    DOI 10.1016/j.commatsci.2017.01.017.  Krížová kontrola: MP mp-219.

    Overené po oprave: 2 atómy, V = 28.644 Å³, min. Mo–P 2.4495 Å (ref. 2.46 Å),
    Mo–Mo 3.190 Å (‖c) a 3.220 Å (bazálne), hustota 7.358 g/cm³
    (exp. 7.34 g/cm³, +0.25 %), ASE potvrdzuje grupu 187.
    """
    from ase.spacegroup import crystal
    return crystal(
        symbols=["Mo", "P"],
        basis=[(0.0, 0.0, 0.0),                    # Mo, Wyckoff 1a
               (1 / 3, 2 / 3, 0.5)],               # P,  Wyckoff 1d
        spacegroup=187,                            # P-6m2 (WC / B_h typ)
        cellpar=[3.22, 3.22, 3.19, 90, 90, 120],
    )


def create_mo2n_bulk():
    """γ-Mo2N: defect rock-salt (B1), Fm-3m (#225), a = 4.16158 Å. OPRAVENÉ 2026-07-28.

    Pôvodná verzia mala 3 atómy (Mo2N1) v CsCl-podobnej Mo podmriežke → spglib ju
    klasifikoval ako P4/mmm (#123, TETRAGONÁLNA), nie kubickú; Mo–Mo 3.603 Å
    namiesto 2.9427 Å (+22.4 %), hustota 4.749 vs exp. 9.496 g/cm³ = PRESNE 2×
    menej. Docstring „cubic anti-perovskite" bol nepravdivý v oboch tvrdeniach.
    Slaby mali doslova polovičnú Mo hustotu a umelé intersticiálne dutiny
    (r = 2.26 vs 2.01 Å), do ktorých H pri DFT relaxácii padal — to je pôvod
    „subpovrchového H" a 5 najhorších DFT odchýlok (Mo2N_(111) −4.85 eV atď.).

    Referencia: COD 1528388 = Bull, C. L. et al., J. Solid State Chem. 179 (2006)
    1762–1767, DOI 10.1016/j.jssc.2006.03.011 (neutrónová + XRD prášková difrakcia):
        Mo1  4a (0, 0, 0)        occ 1.000   → plná fcc podmriežka
        N1   4b (1/2, 1/2, 1/2)  occ 0.506   → N plní POLOVICU oktaédrických dier

    Poloobsadenie sa pre DFT reprezentuje usporiadaním: obsadíme 2 zo 4 miest 4b
    (→ Mo4N2, Z = 2). Overené: všetky 4 neekvivalentné voľby „2 zo 4" dávajú
    identickú hustotu 9.4880 g/cm³ aj identické prvé sféry (Mo–Mo 2.9427,
    Mo–N 2.0808, N–N 2.9427 Å) — voľba ovplyvní len povrchovú termináciu.
    Alternatívy pre skutočne usporiadanú fázu: β-Mo2N I4₁/amd (COD 2311006,
    Evans & Jack, Acta Cryst. 10 (1957) 833) alebo SQS/väčšia superbunka.

    Overené po oprave: 6 atómov (Mo4N2), hustota 9.4880 g/cm³
    (exp. Mo4N2.024 → 9.4958 g/cm³), Mo podmriežka spglib = Fm-3m (#225).
    """
    from ase import Atoms as _Atoms
    from ase.spacegroup import crystal

    # plná fcc Mo podmriežka: 4a → (0,0,0), (0,½,½), (½,0,½), (½,½,0)
    mo = crystal(symbols=["Mo"], basis=[(0.0, 0.0, 0.0)],
                 spacegroup=225, cellpar=[4.16158] * 3 + [90, 90, 90])

    # 2 zo 4 oktaédrických 4b miest → usporiadanie N/vakancia
    n = _Atoms("N2", cell=mo.get_cell(), pbc=True)
    n.set_scaled_positions([(0.5, 0.5, 0.5), (0.0, 0.0, 0.5)])

    return mo + n


def create_mo2c_bulk():
    """α-Mo2C, ξ-Fe2N-typ, ortorombický Pbcn (#60). OPRAVENÉ 2026-07-28.

    Pôvodná hand-coded verzia mala v bunke iba 6 atómov (4 Mo + 2 C), pričom
    Pbcn s Mo na 8d a C na 4c vyžaduje 12 atómov (8 Mo + 4 C). Symetrické
    operácie Pbcn sa NIKDY neaplikovali → spglib klasifikoval výsledok ako
    P2_1 (#4), nie Pbcn (#60). Chýbala polovica mriežky:
      · hustota 4.5553 vs exp. 9.185 g/cm³ = faktor 2.02×
      · v štruktúre zostala UMELÁ DUTINA s polomerom 2.67 Å (reálna 2.10 Å),
        do ktorej pri DFT relaxácii padal H pod povrch → to je pôvod
        „subpovrchového H" a znehodnotilo VŠETKY Mo2C výsledky (39 systémov).
    Wyckoff parametre boli v pôvodnom kóde SPRÁVNE, len neexpandované.

    Referencia: MP mp-1552 / ICSD 43322, dataset DOI 10.17188/1191211;
    mriežkové parametre a Wyckoff podľa arXiv:2201.12706, Tab. 1 a 2
    (experiment a = 4.725, b = 6.022, c = 5.195 Å);
    prototyp indexoval A. N. Christensen, Acta Chem. Scand. A 31 (1977) 509.

    Overené po oprave: 12 atómov (Mo8C4), spglib = Pbcn (#60), min. medziatómová
    vzdialenosť 2.096 Å, hustota 9.185 g/cm³ (RTG referencia 9.19), každý Mo má
    3 C susedov, každý C má 6 Mo susedov (CMo6 oktaédre) — presne ako MP.
    """
    from ase.spacegroup import crystal
    return crystal(
        symbols=["Mo", "C"],
        basis=[(0.250, 0.125, 0.083),     # Mo, Wyckoff 8d
               (0.500, 0.375, 0.250)],    # C,  Wyckoff 4c
        spacegroup=60,                    # Pbcn (ξ-Fe2N-typ, α-Mo2C)
        cellpar=[4.724, 6.004, 5.199, 90, 90, 90],
    )


def create_mob_bulk():
    """Create MoB bulk structure (β-MoB, CrB-type, orthorhombic Cmcm, No. 63).

    OPRAVENÉ 2026-07-23: pôvodná hand-coded I4₁/amd verzia mala PREHODENÉ
    z-súradnice v stĺpcoch (0.5,0.5) a (0.5,0) → Mo a B sa prekrývali na
    0.763 Å (fyzikálne nemožné), čo znehodnotilo VŠETKY MoB DFT aj ML výsledky.
    Nahradené overenou β-MoB (CrB-typ) z Wyckoff pozícií 4c (0, y, 1/4):
    hustota 8.60 g/cm³ (exp. 8.65, <1 % chyba), min. B-B ~1.85 Å (reťazce).
    """
    from ase.spacegroup import crystal
    return crystal(
        symbols=["Mo", "B"],
        basis=[(0.0, 0.146, 0.25), (0.0, 0.440, 0.25)],
        spacegroup=63,                       # Cmcm (CrB-typ)
        cellpar=[3.15, 8.47, 3.09, 90, 90, 90],
    )


def create_ti3c2_bulk():
    """Ti3C2O2 MXene monovrstva, P-3m1 (#164), O na CCP site. OPRAVENÉ 2026-07-28.

    Pôvodná verzia mala UNIFORMNÉ z-frakcie po 0.050 (= 1.000 Å pri c=20 Å) pre
    VŠETKY medzivrstvové vzdialenosti → Ti3C2 jadro stlačené o 13.7 %, stredná
    Ti–C väzba 2.036 Å namiesto 2.202 Å. Bola to „rovnomerná mriežka", nie
    kryštálová štruktúra — z-súradnice neboli odvodené zo žiadnych väzbových dĺžok.

    Nahradené z-frakciami odvodenými z DFT väzbových dĺžok:
    Li, Y. et al., "First-Principles Study on the Structural, Electronic, and
    ... Properties of Ti3C2 MXene", ACS Omega 7 (2022) 40578,
    DOI 10.1021/acsomega.2c05913, Table 1, riadok "Ti3C2O2 / CCP"
    (najstabilnejšia konfigurácia):
        a = 3.083 Å, d(Ti_stred–C) = 2.202, d(Ti_povrch–C) = 2.052,
        d(Ti_povrch–O) = 1.993 Å

    Overené po oprave: spglib potvrdí P-3m1 (#164), Wyckoff ['a','d','d','c'],
    min. vzdialenosť 1.993 Å (Ti–O), Ti–C 2.052 / 2.202 Å, O–O hrúbka 6.428 Å,
    vrstva vycentrovaná na z = 10.0 Å.
    Pozn.: hustota tu NIE JE diagnostikum (bunka je monovrstva + vákuum).
    """
    from ase.spacegroup import crystal
    a, c = 3.083, 20.0
    L = a / np.sqrt(3.0)                        # lateral offset 3-fold hollow = 1.7800 Å
    dz_TiC_mid = np.sqrt(2.202 ** 2 - L ** 2)   # 1.2963 Å
    dz_TiC_surf = np.sqrt(2.052 ** 2 - L ** 2)  # 1.0210 Å
    dz_TiO = np.sqrt(1.993 ** 2 - L ** 2)       # 0.8965 Å
    z_Ti2 = (dz_TiC_mid + dz_TiC_surf) / c                  # 0.11587
    z_C = dz_TiC_mid / c                                    # 0.06482
    z_O = (dz_TiC_mid + dz_TiC_surf + dz_TiO) / c           # 0.16069
    return crystal(
        symbols=["Ti", "Ti", "C", "O"],
        basis=[(0.0, 0.0, 0.0),          # Ti stredné,   Wyckoff 1a
               (1 / 3, 2 / 3, z_Ti2),    # Ti povrchové, Wyckoff 2d
               (2 / 3, 1 / 3, z_C),      # C,            Wyckoff 2d
               (0.0, 0.0, z_O)],         # O na CCP site, Wyckoff 2c
        spacegroup=164,                  # P-3m1
        cellpar=[a, a, c, 90, 90, 120],
    )


def create_graphene_sheet(size=(4, 4, 1), vacuum=10):
    """Create a graphene sheet slab."""
    a = 2.46  # Å, graphene lattice constant

    cell = np.array([
        [a, 0, 0],
        [-a / 2, a * np.sqrt(3) / 2, 0],
        [0, 0, 20.0],
    ])

    symbols = ['C', 'C']
    positions = np.array([
        [0.0, 0.0, 0.5],
        [1 / 3, 2 / 3, 0.5],
    ])

    bulk = Atoms(symbols, cell=cell, pbc=[True, True, True])
    bulk.set_scaled_positions(positions)

    sheet = bulk.repeat(size)
    sheet.set_pbc([True, True, False])
    sheet.center(vacuum=vacuum, axis=2)
    return sheet


def create_n_doped_graphene(size=(4, 4, 1), vacuum=10):
    """Create N-doped graphene (one C replaced by N)."""
    sheet = create_graphene_sheet(size=size, vacuum=vacuum)
    # Replace the C atom closest to center with N
    positions = sheet.get_positions()
    center_xy = np.mean(positions[:, :2], axis=0)
    c_indices = [i for i, atom in enumerate(sheet) if atom.symbol == 'C']
    distances = [np.linalg.norm(positions[i, :2] - center_xy) for i in c_indices]
    replace_idx = c_indices[int(np.argmin(distances))]
    sheet[replace_idx].symbol = 'N'
    return sheet


def _apply_constraints(slab):
    """Freeze bottom half of atoms for stability."""
    z_positions = slab.get_positions()[:, 2]
    z_min = np.min(z_positions)
    z_max = np.max(z_positions)
    z_mid = (z_min + z_max) / 2
    fixed_indices = [i for i in range(len(slab)) if slab[i].z < z_mid]
    slab.set_constraint(FixAtoms(indices=fixed_indices))


def _parse_miller(miller):
    """Convert a string like '(111)' to a Miller index tuple."""
    digits = miller.strip().replace("(", "").replace(")", "")
    if len(digits) != 3 or not digits.isdigit():
        raise ValueError(f"Unsupported Miller index format: {miller}")
    return tuple(int(c) for c in digits)


def _min_same_element_distance(atoms):
    """Najmenšia vzdialenosť medzi atómami TOHO ISTÉHO prvku (PBC-aware)."""
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    best = np.inf
    for sym in set(atoms.get_chemical_symbols()):
        idx = [i for i, a in enumerate(atoms) if a.symbol == sym]
        if len(idx) < 2:
            continue
        best = min(best, min(d[i][j] for i in idx for j in idx if i != j))
    return float(best)


def create_slab(bulk_atoms, miller="(100)", size=(2, 2, 4), vacuum=8):
    """Create a surface slab for a requested Miller index.

    BRÁNA PROTI STACKING FAULTU (pridané 2026-07-28): `ase.build.surface` s
    `layers=size[2]` vytvorí pri niektorých kombináciách rezu a počtu vrstiev
    AA stacking fault — dva atómy toho istého prvku skončia bližšie než v bulku.
    OVERENÉ na γ-Mo2N(111): `layers=4` (default!) dá Mo–Mo 2.4027 Å namiesto
    2.9427 Å, kým layers 3/5/6/8 sú v poriadku; (100) a (110) sú v poriadku vždy.
    Predtým by taký slab prešiel ticho a znehodnotil celú fazetu.

    Preto: postav slab, porovnaj min. rovnako-prvkovú vzdialenosť s bulkom a ak
    je degradovaná o >5 %, skús layers+1 … layers+3. Ak sa to nepodarí, vyhoď
    výnimku — NIKDY nevracaj rozbitú geometriu.
    """
    indices = _parse_miller(miller)
    ref = _min_same_element_distance(bulk_atoms)

    last_err = None
    for extra in (0, 1, 2, 3):
        layers = size[2] + extra
        slab = surface(bulk_atoms, indices, layers=layers, vacuum=vacuum, periodic=True)
        slab = slab.repeat((size[0], size[1], 1))
        slab.set_pbc([True, True, False])
        slab.center(vacuum=vacuum, axis=2)
        got = _min_same_element_distance(slab)
        if not np.isfinite(ref) or got >= 0.95 * ref:
            if extra:
                print(f"    ⓘ {miller}: layers {size[2]}→{layers} "
                      f"(pôvodné dávalo stacking fault {last_err:.4f} Å < {0.95*ref:.4f} Å)")
            _apply_constraints(slab)
            return slab
        last_err = got

    raise ValueError(
        f"create_slab({miller}, size={size}): stacking fault sa nedá odstrániť ani "
        f"pri layers={size[2]}+3 — min. rovnako-prvková vzdialenosť {last_err:.4f} Å "
        f"vs bulk {ref:.4f} Å. Skontroluj bulk štruktúru a rez.")


def create_edge_ribbon(bulk_atoms, width=6, length=2, vacuum=8, edge_type="Mo"):
    """Create a simple edge ribbon by adding vacuum in x and z.

    edge_type:
        "Mo"  -> remove chalcogen atoms at ribbon edges
        "X"   -> remove Mo atoms at ribbon edges (X = S or Se)
    """
    ribbon = bulk_atoms.repeat((width, length, 1))

    # Create vacuum in x and z to form edges and a single-layer ribbon
    ribbon.center(vacuum=vacuum, axis=0)
    ribbon.center(vacuum=vacuum, axis=2)
    ribbon.set_pbc([False, True, False])

    # Determine edge atoms by x position
    positions = ribbon.get_positions()
    x_positions = positions[:, 0]
    x_min = np.min(x_positions)
    x_max = np.max(x_positions)
    tol = 0.3  # Angstroms
    edge_mask = (x_positions - x_min < tol) | (x_max - x_positions < tol)

    # Remove atoms at edges to approximate termination
    if edge_type == "Mo":
        remove_symbols = {"S", "Se"}
    else:
        remove_symbols = {"Mo"}

    remove_indices = [
        i for i, atom in enumerate(ribbon)
        if edge_mask[i] and atom.symbol in remove_symbols
    ]
    if remove_indices:
        del ribbon[remove_indices]

    _apply_constraints(ribbon)
    return ribbon


class SiteSelectionError(RuntimeError):
    """Raised when no valid surface site exists for a defect/dopant request."""


# Maximum depth below the global surface (Å) at which a site still counts as
# "surface". One interlayer spacing is enough for all facets used here.
SURFACE_DEPTH_TOL = 2.5


def _species_top_layer(slab, symbol, tol=0.5, min_count=1,
                       max_depth=SURFACE_DEPTH_TOL):
    """Indices of `symbol` atoms in the topmost layer *of that species*.

    OPRAVA (2026-07-28): pôvodná verzia hľadala atómy do 0.5 Å od GLOBÁLNEHO
    z_max. Na termináciách, kde je požadovaný prvok hoc len o 0.05 Å nižšie ako
    najvyššia vrstva iného prvku (Mo2C(111): Mo je 0.545 Å pod C), nenašla NIČ
    a volajúci ticho vrátil pristinný slab → dáta bez vakancie/dopantu.
    Teraz sa referencia berie z maxima z-súradnice DANÉHO prvku a nedostatok
    kandidátov je HLASNÁ chyba, nie tichý no-op.
    """
    z = slab.get_positions()[:, 2]
    idx = [i for i in range(len(slab)) if slab[i].symbol == symbol]
    if not idx:
        raise SiteSelectionError(f"slab contains no {symbol} atoms at all")

    global_top = float(np.max(z))
    species_top = float(np.max(z[idx]))
    depth = global_top - species_top
    if depth > max_depth:
        raise SiteSelectionError(
            f"topmost {symbol} atom is {depth:.2f} Å below the surface "
            f"(> {max_depth} Å) – not a surface site on this termination"
        )

    layer = [i for i in idx if species_top - z[i] < tol]
    # If the species' top layer is too sparse for the requested count, extend
    # to the next-deepest layer(s) of the same species instead of failing.
    while len(layer) < min_count:
        rest = [i for i in idx if i not in layer]
        if not rest:
            raise SiteSelectionError(
                f"only {len(layer)} {symbol} atoms available, need {min_count}"
            )
        next_top = float(np.max(z[rest]))
        if (global_top - next_top) > max_depth:
            raise SiteSelectionError(
                f"only {len(layer)} surface {symbol} atoms available, "
                f"need {min_count}"
            )
        layer += [i for i in rest if next_top - z[i] < tol]
    return layer


def _sort_by_center(slab, indices):
    """Order indices by in-plane distance from the slab centre."""
    positions = slab.get_positions()
    center_xy = np.mean(positions[:, :2], axis=0)
    dist = {i: float(np.linalg.norm(positions[i, :2] - center_xy)) for i in indices}
    return sorted(indices, key=lambda i: dist[i])


def create_vacancy_slab(slab, vacancy_symbol):
    """Remove one top-layer atom of `vacancy_symbol` to create a vacancy."""
    n_before = len(slab)
    candidates = _species_top_layer(slab, vacancy_symbol, min_count=1)
    remove_index = _sort_by_center(slab, candidates)[0]
    del slab[remove_index]
    assert len(slab) == n_before - 1, "vacancy was not created"
    return slab


def create_multi_vacancy_slab(slab, vacancy_symbol, count=2):
    """Remove `count` top-layer atoms to create a vacancy cluster."""
    n_before = len(slab)
    candidates = _species_top_layer(slab, vacancy_symbol, min_count=count)
    remove_indices = _sort_by_center(slab, candidates)[:count]
    for idx in sorted(remove_indices, reverse=True):
        del slab[idx]
    assert len(slab) == n_before - count, "vacancy cluster was not created"
    return slab


def create_substitution_slab(slab, target_symbol, dopant_symbol):
    """Substitute one top-layer `target_symbol` atom with a dopant."""
    candidates = _species_top_layer(slab, target_symbol, min_count=1)
    replace_index = _sort_by_center(slab, candidates)[0]
    slab[replace_index].symbol = dopant_symbol
    assert dopant_symbol in slab.get_chemical_symbols(), "dopant was not inserted"
    return slab


def add_cluster_on_surface(slab, element, n_atoms=2, height=1.8, spacing=2.4):
    """Add a small cluster (2 or 4 atoms) above the top surface."""
    positions = slab.get_positions()
    z_max = np.max(positions[:, 2])
    center_xy = np.mean(positions[:, :2], axis=0)

    if n_atoms == 2:
        offsets = [(-spacing / 2, 0.0), (spacing / 2, 0.0)]
    elif n_atoms == 4:
        offsets = [
            (-spacing / 2, -spacing / 2),
            (-spacing / 2, spacing / 2),
            (spacing / 2, -spacing / 2),
            (spacing / 2, spacing / 2),
        ]
    else:
        offsets = [(0.0, 0.0)]

    for dx, dy in offsets:
        slab += Atoms(element, positions=[[center_xy[0] + dx, center_xy[1] + dy, z_max + height]])
    return slab


def create_ni_slab(miller="(111)", size=(3, 3, 4), vacuum=8):
    """Create a simple Ni slab for interface models."""
    a = 3.52  # Angstrom, fcc Ni
    if miller == "(100)":
        slab = fcc100("Ni", size=size, a=a, vacuum=vacuum)
    else:
        slab = fcc111("Ni", size=size, a=a, vacuum=vacuum)
    slab.set_pbc([True, True, False])
    return slab


def create_ni_mox_interface(mox_bulk_builder, miller="(111)", separation=2.2):
    """Create a simple Ni/MoX interface slab (trend-level model).

    Args:
        mox_bulk_builder: callable returning bulk Atoms (e.g. create_mo2n_bulk)
        miller: Miller index string
        separation: gap between Ni top and MoX bottom (Å)
    """
    ni = create_ni_slab(miller=miller, size=(3, 3, 4), vacuum=8)
    mox_bulk = mox_bulk_builder()
    mox = create_slab(mox_bulk, miller=miller, size=(2, 2, 2), vacuum=0)
    mox.set_pbc([True, True, False])

    # POZOR (2026-07-23): tento „trend-level" model NEzosúlaďuje mriežky Ni(3×3)
    # a MoX(2×2) — sú nekomenzurabilné (líšia sa 2–3×). Po `set_cell(ni.cell)` sa
    # MoX atómy zabalia cez PBC a prekryjú (najmä (111): min-dist 0.43–0.92 Å).
    # Epitaxný strain na Ni bunku vyžaduje −47..−71 % (nefyzikálne). Preto sú tieto
    # interface geometricky nevalidné a NEPOUŽÍVAJÚ sa v korigovanej validácii;
    # správne riešenie by bola komenzurabilná superbunka (samostatná úloha).

    # Stack MoX on top of Ni
    ni_positions = ni.get_positions()
    ni_top = np.max(ni_positions[:, 2])

    mox_positions = mox.get_positions()
    mox_shift = ni_top + separation - np.min(mox_positions[:, 2])
    mox.translate([0.0, 0.0, mox_shift])

    # Define combined cell with additional vacuum
    cell = ni.cell.copy()
    cell[2, 2] = np.max(mox.get_positions()[:, 2]) + 8.0

    interface = ni + mox
    interface.set_cell(cell)
    interface.set_pbc([True, True, False])

    # Freeze bottom half of Ni atoms only
    ni_indices = [i for i, atom in enumerate(interface) if atom.symbol == "Ni"]
    ni_z = interface.get_positions()[ni_indices, 2]
    ni_z_mid = (np.min(ni_z) + np.max(ni_z)) / 2
    fixed_indices = [i for i in ni_indices if interface[i].z < ni_z_mid]
    interface.set_constraint(FixAtoms(indices=fixed_indices))
    return interface


# Keep the old name as an alias for backward compatibility
def create_ni_mo2n_interface(miller="(111)", separation=2.2):
    """Create Ni/Mo2N interface (backward-compatible wrapper)."""
    return create_ni_mox_interface(create_mo2n_bulk, miller=miller, separation=separation)


def create_ni_mxene_interface(miller="(111)", separation=2.2):
    """Create Ni on Ti3C2O2 MXene interface."""
    ni = create_ni_slab(miller=miller, size=(3, 3, 4), vacuum=8)
    mxene = create_ti3c2_bulk()
    # Use a 3x3 supercell of MXene for size-matching with Ni slab
    mxene = mxene.repeat((3, 3, 1))
    mxene.set_pbc([True, True, False])

    ni_positions = ni.get_positions()
    ni_top = np.max(ni_positions[:, 2])

    mxene_positions = mxene.get_positions()
    mxene_shift = ni_top + separation - np.min(mxene_positions[:, 2])
    mxene.translate([0.0, 0.0, mxene_shift])

    cell = ni.cell.copy()
    cell[2, 2] = np.max(mxene.get_positions()[:, 2]) + 8.0

    interface = ni + mxene
    interface.set_cell(cell)
    interface.set_pbc([True, True, False])

    ni_indices = [i for i, atom in enumerate(interface) if atom.symbol == "Ni"]
    ni_z = interface.get_positions()[ni_indices, 2]
    ni_z_mid = (np.min(ni_z) + np.max(ni_z)) / 2
    fixed_indices = [i for i in ni_indices if interface[i].z < ni_z_mid]
    interface.set_constraint(FixAtoms(indices=fixed_indices))
    return interface


def create_ni_on_graphene(ni_atoms=4, height=1.8, size=(4, 4, 1), vacuum=10):
    """Create Ni cluster on graphene sheet."""
    sheet = create_graphene_sheet(size=size, vacuum=vacuum)
    sheet = add_cluster_on_surface(sheet, "Ni", n_atoms=ni_atoms, height=height)
    _apply_constraints(sheet)
    return sheet


def create_ni_on_n_doped_graphene(ni_atoms=4, height=1.8, size=(4, 4, 1), vacuum=10):
    """Create Ni cluster on N-doped graphene sheet."""
    sheet = create_n_doped_graphene(size=size, vacuum=vacuum)
    sheet = add_cluster_on_surface(sheet, "Ni", n_atoms=ni_atoms, height=height)
    _apply_constraints(sheet)
    return sheet


def create_graphene_nanoribbon(width=6, length=3, vacuum=8):
    """Create armchair graphene nanoribbon (CNT-like approximation)."""
    a = 2.46
    cell = np.array([
        [a, 0, 0],
        [-a / 2, a * np.sqrt(3) / 2, 0],
        [0, 0, 20.0],
    ])
    symbols = ['C', 'C']
    positions = np.array([
        [0.0, 0.0, 0.5],
        [1 / 3, 2 / 3, 0.5],
    ])
    bulk = Atoms(symbols, cell=cell, pbc=[True, True, True])
    bulk.set_scaled_positions(positions)

    ribbon = bulk.repeat((width, length, 1))
    ribbon.center(vacuum=vacuum, axis=0)
    ribbon.center(vacuum=vacuum, axis=2)
    ribbon.set_pbc([False, True, False])
    _apply_constraints(ribbon)
    return ribbon


def create_interface_with_dopant_generic(mox_bulk_builder, miller, dopant, target_symbol="Ni"):
    """Create Ni/MoX interface with a single dopant on the Ni top layer."""
    interface = create_ni_mox_interface(mox_bulk_builder, miller=miller)
    interface = create_substitution_slab(interface, target_symbol, dopant)
    return interface


def create_interface_with_cluster_generic(mox_bulk_builder, miller, dopant, cluster_size):
    """Create Ni/MoX interface with a small dopant cluster on surface."""
    interface = create_ni_mox_interface(mox_bulk_builder, miller=miller)
    interface = add_cluster_on_surface(interface, dopant, n_atoms=cluster_size)
    return interface


def create_interface_with_dopant(miller, dopant, target_symbol="Ni"):
    """Create Ni/Mo2N interface with a single dopant on the Ni top layer."""
    interface = create_ni_mo2n_interface(miller=miller)
    interface = create_substitution_slab(interface, target_symbol, dopant)
    return interface


def create_interface_with_cluster(miller, dopant, cluster_size):
    """Create Ni/Mo2N interface with a small dopant cluster on Ni surface."""
    interface = create_ni_mo2n_interface(miller=miller)
    interface = add_cluster_on_surface(interface, dopant, n_atoms=cluster_size)
    return interface


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
# Overridable so the generator can be dry-run into a scratch tree without
# touching production inputs.
DATA_INPUTS = Path(os.environ.get(
    "CEMEA_VASP_INPUTS", REPO_ROOT / "data" / "inputs" / "VASP_inputs"))

FAILED_STRUCTURES = []


def generate_all_structures():
    """Generate all structure files"""
    
    # Map formulas to builder functions
    builders = {
        'MoS2': create_mos2_bulk,
        'MoSe2': create_mose2_bulk,
        'MoP': create_mop_bulk,
        'Mo2N': create_mo2n_bulk,
        'Mo2C': create_mo2c_bulk,
        'MoB': create_mob_bulk,
    }

    # Chalcogenide-like compounds that get vacancy/edge/sheet variants
    chalcogenides = {
        'MoS2': 'S',
        'MoSe2': 'Se',
    }

    # Compounds that get dopant/vacancy treatment (metal sublattice + anion)
    dopant_compounds = {
        'Mo2N': {'metal': 'Mo', 'anion': 'N'},
        'Mo2C': {'metal': 'Mo', 'anion': 'C'},
        'MoB':  {'metal': 'Mo', 'anion': 'B'},
    }

    # Systems that get Ni/MoX interface treatment
    interface_systems = {
        'Ni_Mo2N': create_mo2n_bulk,
        'Ni_Mo2C': create_mo2c_bulk,
        'Ni_MoB':  create_mob_bulk,
        'Ni_MoS2': create_mos2_bulk,
    }

    dopants = ["Pt", "Pd", "Ir", "Ru", "Ag", "Au", "Ni"]
    # Anna's decoration list (subset for interfaces)
    decorations = ["Ag", "Au", "Pd", "Pt", "Ir"]
    
    millers = ['(100)', '(110)', '(111)']
    interface_millers = ['(111)', '(100)']
    
    print("\n" + "="*60)
    print("Generating Structure Files for GPAW Calculations")
    print("="*60)
    
    base_dir = DATA_INPUTS
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Part 1: Basic slabs for all compounds ────────────────────
    for formula, builder in builders.items():
        print(f"\n{formula}:")
        
        try:
            bulk = builder()
            print(f"  Bulk structure: {len(bulk)} atoms/cell")
            
            for miller in millers:
                dir_name = base_dir / f"{formula}_{miller}"
                dir_name.mkdir(parents=True, exist_ok=True)
                poscar_file = dir_name / "POSCAR"
                print(f"    {miller}: ", end="", flush=True)
                try:
                    slab = create_slab(bulk.copy(), miller=miller)
                    write(str(poscar_file), slab, format='vasp')
                    print(f"✓ ({len(slab)} atoms)")
                except Exception as e:
                    print(f"✗ Error: {e}")
        except Exception as e:
            print(f"  ✗ Failed to create {formula}: {e}")

    # ── Part 2: Chalcogenide variants (vacancies, edges, sheets) ─
    for formula, vac_sym in chalcogenides.items():
        print(f"\n{formula} variants:")
        bulk = builders[formula]()

        for miller in millers:
            # Single vacancy
            _write_structure(base_dir, f"{formula}_{miller}_vac{vac_sym}",
                lambda m=miller: create_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), vac_sym))
            # Double vacancy
            _write_structure(base_dir, f"{formula}_{miller}_vac2{vac_sym}",
                lambda m=miller: create_multi_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), vac_sym, count=2))
            # Nanosheet
            _write_structure(base_dir, f"{formula}_{miller}_sheet",
                lambda m=miller: create_slab(bulk.copy(), miller=m, size=(4, 4, 4), vacuum=10))

        # Edge ribbons
        for edge_type, label in [("Mo", "Mo"), ("X", vac_sym)]:
            _write_structure(base_dir, f"{formula}_edge_{label}",
                lambda et=edge_type: create_edge_ribbon(bulk.copy(), edge_type=et))
            _write_structure(base_dir, f"{formula}_edge_{label}_large",
                lambda et=edge_type: create_edge_ribbon(bulk.copy(), width=10, length=3, edge_type=et))

    # ── Part 3: Dopant/vacancy compounds (Mo2N, Mo2C, MoB) ──────
    for formula, info in dopant_compounds.items():
        print(f"\n{formula} dopants/vacancies:")
        bulk = builders[formula]()
        metal, anion = info['metal'], info['anion']

        for miller in millers:
            # Anion vacancy
            _write_structure(base_dir, f"{formula}_{miller}_vac{anion}",
                lambda m=miller: create_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), anion))
            # Metal vacancy
            _write_structure(base_dir, f"{formula}_{miller}_vac{metal}",
                lambda m=miller: create_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), metal))
            # Double vacancies
            _write_structure(base_dir, f"{formula}_{miller}_vac2{anion}",
                lambda m=miller: create_multi_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), anion, count=2))
            _write_structure(base_dir, f"{formula}_{miller}_vac2{metal}",
                lambda m=miller: create_multi_vacancy_slab(
                    create_slab(bulk.copy(), miller=m), metal, count=2))
            # Metal-site dopants
            for dopant in dopants:
                _write_structure(base_dir, f"{formula}_{miller}_dop{dopant}",
                    lambda m=miller, d=dopant: create_substitution_slab(
                        create_slab(bulk.copy(), miller=m), metal, d))

        # Edge ribbons for Mo2C and MoB
        if formula in ('Mo2C', 'MoB'):
            for edge_type, label in [("Mo", "Mo"), ("X", anion)]:
                _write_structure(base_dir, f"{formula}_edge_{label}",
                    lambda et=edge_type: create_edge_ribbon(bulk.copy(), edge_type=et))
                _write_structure(base_dir, f"{formula}_edge_{label}_large",
                    lambda et=edge_type: create_edge_ribbon(bulk.copy(), width=10, length=3, edge_type=et))

            # Nanosheet
            for miller in millers:
                _write_structure(base_dir, f"{formula}_{miller}_sheet",
                    lambda m=miller: create_slab(bulk.copy(), miller=m, size=(4, 4, 4), vacuum=10))

    # ── Part 4: Ni/MoX interfaces ────────────────────────────────
    for sys_name, mox_builder in interface_systems.items():
        print(f"\n{sys_name} interfaces:")

        for miller in interface_millers:
            # Pristine interface
            _write_structure(base_dir, f"{sys_name}_interface_{miller}",
                lambda m=miller, b=mox_builder: create_ni_mox_interface(b, miller=m))

            # Single-atom dopants on Ni layer
            for dopant in dopants:
                _write_structure(base_dir, f"{sys_name}_interface_{miller}_dop{dopant}",
                    lambda m=miller, b=mox_builder, d=dopant:
                        create_interface_with_dopant_generic(b, m, d))

            # Noble metal clusters (2 and 4 atoms)
            for dec in decorations:
                for cs in [2, 4]:
                    _write_structure(base_dir, f"{sys_name}_interface_{miller}_cluster{cs}{dec}",
                        lambda m=miller, b=mox_builder, d=dec, s=cs:
                            create_interface_with_cluster_generic(b, m, d, s))

    # ── Part 5: MXene Ti3C2 + Ni ─────────────────────────────────
    print("\nNi/MXene Ti3C2:")

    # Bare MXene slabs
    mxene_bulk = create_ti3c2_bulk()
    for miller in millers:
        _write_structure(base_dir, f"Ti3C2O2_{miller}",
            lambda m=miller: create_slab(mxene_bulk.copy(), miller=m))

    # Ni/MXene interfaces
    for miller in interface_millers:
        _write_structure(base_dir, f"Ni_Ti3C2O2_interface_{miller}",
            lambda m=miller: create_ni_mxene_interface(miller=m))
        # Decorations on Ni/MXene
        for dec in decorations:
            for cs in [2, 4]:
                _write_structure(base_dir, f"Ni_Ti3C2O2_interface_{miller}_cluster{cs}{dec}",
                    lambda m=miller, d=dec, s=cs: add_cluster_on_surface(
                        create_ni_mxene_interface(miller=m), d, n_atoms=s))

    # ── Part 6: Ni on carbon variants ────────────────────────────
    print("\nNi on carbon:")

    # Pristine graphene
    _write_structure(base_dir, "graphene_sheet",
        lambda: create_graphene_sheet())

    # N-doped graphene
    _write_structure(base_dir, "graphene_N_doped",
        lambda: create_n_doped_graphene())

    # Graphene nanoribbon (CNT approximation)
    _write_structure(base_dir, "graphene_nanoribbon",
        lambda: create_graphene_nanoribbon())

    # Ni on graphene
    for n_ni in [2, 4]:
        _write_structure(base_dir, f"Ni{n_ni}_on_graphene",
            lambda n=n_ni: create_ni_on_graphene(ni_atoms=n))
        _write_structure(base_dir, f"Ni{n_ni}_on_graphene_N_doped",
            lambda n=n_ni: create_ni_on_n_doped_graphene(ni_atoms=n))

    # Ni on nanoribbon
    _write_structure(base_dir, "Ni4_on_nanoribbon",
        lambda: add_cluster_on_surface(
            create_graphene_nanoribbon(), "Ni", n_atoms=4))

    print("\n" + "="*60)
    print("✓ Structure generation complete!")
    if FAILED_STRUCTURES:
        print(f"✗ {len(FAILED_STRUCTURES)} structures SKIPPED (no POSCAR written):")
        for name, err in FAILED_STRUCTURES:
            print(f"    {name}: {err}")
    print("="*60)
    return FAILED_STRUCTURES


def _write_structure(base_dir, name, builder_fn):
    """Helper: build a structure and write its POSCAR.

    On failure NO POSCAR is written and any pre-existing (possibly stale /
    silently-pristine) file is removed, so a broken structure can never be
    mistaken for a valid one downstream.
    """
    dir_name = base_dir / name
    dir_name.mkdir(parents=True, exist_ok=True)
    poscar_file = dir_name / "POSCAR"
    print(f"    {name}: ", end="", flush=True)
    try:
        slab = builder_fn()
        write(str(poscar_file), slab, format='vasp')
        print(f"✓ ({len(slab)} atoms, {slab.get_chemical_formula()})")
    except Exception as e:
        if poscar_file.exists():
            poscar_file.unlink()
        FAILED_STRUCTURES.append((name, str(e)))
        print(f"✗ SKIPPED: {e}")


if __name__ == '__main__':
    generate_all_structures()
    print("\n✓ Done! All POSCAR files created.")
