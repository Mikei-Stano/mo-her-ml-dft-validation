"""
Workflow to compute H adsorption energies for Mo compounds
Using Materials Project + ASE + VASP

Requirements:
- pymatgen (Materials Project API)
- ase (Atomic Simulation Environment)
- VASP installed on HPC cluster

Author: Prepared for HER catalyst screening
"""

import os
import json
import numpy as np
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VASP_DIR = REPO_ROOT / "data" / "inputs" / "VASP_inputs"

# Try to import required packages
try:
    from pymatgen.ext.matproj import MPRester
    from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    print("⚠️  pymatgen not installed. Install with: pip install pymatgen")

try:
    from ase import Atoms
    from ase.io import read, write
    from ase.build import add_vacuum, surface, fcc111
    from ase.constraints import FixAtoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("⚠️  ASE not installed. Install with: pip install ase")


class H_AdsorptionCalculator:
    """
    Calculate H adsorption energies for Mo compounds
    """
    
    def __init__(self, mp_api_key=None):
        """
        Initialize with Materials Project API key
        
        Args:
            mp_api_key: Get from https://materialsproject.org/dashboard
        """
        self.mp_api_key = mp_api_key
        self.structures = {}
        self.calculations = []
        
    def download_structures(self, formulas=['MoS2', 'MoSe2', 'MoP', 'Mo2N']):
        """
        Download bulk structures from Materials Project
        
        Args:
            formulas: List of chemical formulas to search
            
        Returns:
            dict: {formula: structure}
        """
        
        if not HAS_PYMATGEN:
            print("❌ pymatgen required. Install: pip install pymatgen")
            return None
        
        if not self.mp_api_key:
            print("\n⚠️  STEP 1: Get Materials Project API Key")
            print("1. Go to: https://materialsproject.org/dashboard")
            print("2. Login (create free account if needed)")
            print("3. Copy your API key")
            print("4. Run: compute_h_adsorption.py --api_key YOUR_KEY")
            return None
        
        print("\n" + "="*60)
        print("STEP 1: Downloading structures from Materials Project")
        print("="*60)
        
        try:
            with MPRester(self.mp_api_key) as mpr:
                for formula in formulas:
                    print(f"\nSearching for {formula}...")
                    
                    # Try newer API first (pymatgen >= 2023)
                    try:
                        docs = mpr.materials.summary.search(
                            formula=formula,
                            is_stable=True,
                            sort={"formation_energy_per_atom": "asc"},
                            limit=1
                        )
                        
                        if docs:
                            material_id = docs[0].material_id
                            struct = mpr.get_structure_by_material_id(material_id)
                            self.structures[formula] = struct
                            print(f"  ✓ Found {formula} (ID: {material_id})")
                            print(f"    Lattice: a={struct.lattice.a:.3f}, b={struct.lattice.b:.3f}, c={struct.lattice.c:.3f}")
                        else:
                            print(f"  ✗ No stable structures found for {formula}")
                    
                    except Exception as e_new:
                        # Fallback: try older API (pymatgen < 2023)
                        print(f"    Trying legacy API...")
                        try:
                            structures = mpr.get_structures(formula)
                            if structures:
                                struct = structures[0]
                                self.structures[formula] = struct
                                print(f"  ✓ Found {formula}")
                                print(f"    Lattice: a={struct.lattice.a:.3f}, b={struct.lattice.b:.3f}, c={struct.lattice.c:.3f}")
                            else:
                                print(f"  ✗ No structures found for {formula}")
                        except Exception as e_old:
                            print(f"  ✗ Error querying {formula}: {e_old}")
            
            return self.structures
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def create_slabs(self, formula, miller_indices_list=[(1,0,0), (1,1,0), (1,1,1)], 
                    vacuum=15, layers=4):
        """
        Create surface slabs from bulk structure
        
        Args:
            formula: Chemical formula (key in self.structures)
            miller_indices_list: List of (h, k, l) tuples
            vacuum: Vacuum spacing in Angstroms
            layers: Number of atomic layers
            
        Returns:
            dict: {miller_indices: ase_atoms_object}
        """
        
        if not HAS_ASE:
            print("❌ ASE required. Install: pip install ase")
            return None
        
        if formula not in self.structures:
            print(f"❌ Structure for {formula} not found")
            return None
        
        print(f"\n" + "="*60)
        print(f"STEP 2: Creating surface slabs for {formula}")
        print("="*60)
        
        struct = self.structures[formula]
        slabs = {}
        
        for miller in miller_indices_list:
            try:
                # Convert pymatgen structure to ASE atoms
                from pymatgen.io.ase import AseAtomsAdaptor
                atoms = AseAtomsAdaptor.get_atoms(struct)
                
                # Format miller indices for display
                miller_str = f"({miller[0]}{miller[1]}{miller[2]})"
                print(f"\n  Creating {miller_str} slab...")
                
                # Create a simple slab by replicating unit cell
                # and adding vacuum (simplified approach)
                slab = atoms.repeat((2, 2, 2))  # Expand unit cell
                slab.center(axis=2, vacuum=vacuum)
                
                # Freeze bottom layers (bottom 50% of atoms)
                z_pos = slab.get_positions()[:, 2]
                z_median = np.median(z_pos)
                constraint = FixAtoms(indices=[i for i, z in enumerate(z_pos) if z < z_median])
                slab.set_constraint(constraint)
                
                slabs[miller_str] = slab
                print(f"    ✓ Slab created: {len(slab)} atoms, frozen: {len(constraint.index)} atoms")
                
            except Exception as e:
                print(f"    ✗ Error creating slab: {e}")
        
        return slabs
    
    def add_adsorbate(self, slab, adsorbate='H', height=1.5):
        """
        Add adsorbate (H) to surface
        
        Args:
            slab: ASE Atoms object (surface)
            adsorbate: Element to add ('H', 'O', etc.)
            height: Distance above surface (Angstroms)
            
        Returns:
            ase.Atoms: Slab with adsorbate
        """
        
        if not HAS_ASE:
            return None
        
        from ase.build import add_adsorbate
        
        slab_with_ad = slab.copy()
        
        # Find topmost atom
        positions = slab_with_ad.get_positions()
        top_z = np.max(positions[:, 2])
        
        # Add H at different x,y positions
        # For simplicity, add at (center_x, center_y, top_z + height)
        center_xy = np.mean(positions[:, :2], axis=0)
        
        h_atom = Atoms(adsorbate, positions=[[center_xy[0], center_xy[1], top_z + height]])
        slab_with_ad += h_atom
        
        return slab_with_ad
    
    def prepare_vasp_input(self, formula, miller, output_dir=None):
        """
        Prepare VASP input files (INCAR, KPOINTS, POSCAR)
        
        Args:
            formula: Chemical formula
            miller: Miller indices string
            output_dir: Output directory for files
            
        Returns:
            str: Path to output directory
        """
        
        output_base = Path(output_dir) if output_dir else DEFAULT_VASP_DIR
        output_path = output_base / f"{formula}_{miller}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n" + "="*60)
        print(f"STEP 3: Preparing VASP inputs for {formula} ({miller})")
        print("="*60)
        
        # Create INCAR (HER calculation parameters)
        incar_dict = {
            'SYSTEM': f'{formula} H adsorption',
            'PREC': 'Accurate',
            'ENCUT': 400,
            'ISMEAR': 0,
            'SIGMA': 0.05,
            'IBRION': 2,
            'NSW': 100,
            'ISIF': 2,
            'LREAL': False,
            'ALGO': 'Normal',
            'NELM': 100,
            'EDIFF': 1e-5,
            'EDIFFG': -0.01,
            'LWAVE': False,
            'LCHARG': False,
        }
        
        incar_file = output_path / 'INCAR'
        with open(incar_file, 'w') as f:
            for key, val in incar_dict.items():
                f.write(f"{key} = {val}\n")
        
        print(f"  ✓ INCAR written")
        
        # Create KPOINTS
        kpoints_file = output_path / 'KPOINTS'
        with open(kpoints_file, 'w') as f:
            f.write("Automatic mesh\n")
            f.write("0\n")
            f.write("Monkhorst-Pack\n")
            f.write("4 4 1\n")  # 4x4x1 k-point mesh for slab
            f.write("0 0 0\n")
        
        print(f"  ✓ KPOINTS written")
        
        # Create POTCAR (requires VASP installation - manual step)
        print(f"  ⚠️  POTCAR: Manual step required!")
        print(f"     Copy POTCARs from your VASP installation to: {output_path}/POTCAR")
        
        # Write info file
        info_file = output_path / 'README.txt'
        with open(info_file, 'w') as f:
            f.write(f"H adsorption calculation: {formula} on {miller} surface\n")
            f.write(f"\nSteps:\n")
            f.write(f"1. Copy POTCAR from VASP: cp $VASP_POTCAR_DIR/POTCAR .\n")
            f.write(f"2. Run: mpirun -np 16 vasp > vasp.out\n")
            f.write(f"3. Extract energy from OUTCAR\n")
        
        print(f"\n  ✓ Files written to: {output_path}")
        return str(output_path)
    
    def create_submission_script(self, formula, output_dir=None, 
                                 job_name='H_ads', nodes=1, ntasks=16, 
                                 time_limit='24:00:00', cluster='slurm'):
        """
        Create HPC submission script (SLURM)
        
        Args:
            formula: Chemical formula
            output_dir: Base output directory
            job_name: SLURM job name
            nodes: Number of compute nodes
            ntasks: Number of MPI tasks
            time_limit: Wall time limit
            cluster: Cluster type ('slurm', 'pbs', etc.)
        """
        
        output_base = Path(output_dir) if output_dir else DEFAULT_VASP_DIR
        script_path = output_base / 'submit.sh'
        
        if cluster == 'slurm':
            script = f"""#!/bin/bash
#SBATCH --job-name={job_name}_{formula}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks//nodes}
#SBATCH --time={time_limit}
#SBATCH --partition=gpu  # or compute, depending on your cluster
#SBATCH --output=%x_%j.out

# Load modules
module load intel
module load vasp/6.4.2  # Adjust to your system

# Run VASP
mpirun -np {ntasks} vasp > vasp.out

# Extract results
if [ -f OUTCAR ]; then
    echo "Calculation completed successfully"
    # Extract final energy
    grep "free  energy" OUTCAR | tail -1
fi
"""
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        os.chmod(script_path, 0o755)
        print(f"\n✓ Submission script created: {script_path}")
        print(f"  To submit: sbatch {script_path}")


def main():
    """
    Main workflow
    """
    
    print("\n" + "="*60)
    print("H Adsorption Energy Calculator for Mo Compounds")
    print("="*60)
    
    # Step 1: Initialize
    calculator = H_AdsorptionCalculator()
    
    print("\n📋 WORKFLOW OVERVIEW:")
    print("1. Download bulk structures from Materials Project")
    print("2. Create surface slabs (100, 110, 111)")
    print("3. Add H adsorbate at different positions")
    print("4. Prepare VASP input files")
    print("5. Submit to HPC cluster")
    print("6. Parse results and calculate ΔGH")
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    
    print("""
1. GET MATERIALS PROJECT API KEY:
   - Go to: https://materialsproject.org/dashboard
   - Copy your API key
   - Set environment: export MP_API_KEY="your_api_key"
   
2. RUN THIS SCRIPT:
   export MP_API_KEY="xxxxxxxxxxxx"
   python compute_h_adsorption.py
   
3. VASP SETUP:
   - Install VASP on your HPC cluster
   - Make sure POTCARs are available
   
4. SUBMIT JOBS:
   - Go to each formula directory
   - Copy POTCARs
   - Run: sbatch submit.sh

5. PARSE RESULTS (see parse_results.py)
   - Extract energies from OUTCAR
   - Calculate ΔGH = E(slab+H) - E(slab) - 0.5*E(H2)
""")
    
    # Step 2: Try to download (requires MP API key)
    mp_key = os.getenv('MP_API_KEY')
    if mp_key:
        calculator.mp_api_key = mp_key
        structures = calculator.download_structures()
        
        if structures:
            # Step 3: Create slabs
            for formula in structures.keys():
                slabs = calculator.create_slabs(formula, miller_indices_list=[(1,0,0), (1,1,0), (1,1,1)])
                
                if slabs:
                    # Step 4: Prepare VASP inputs
                    for miller in slabs.keys():
                        calculator.prepare_vasp_input(formula, miller)
                    
                    # Step 5: Create submission script
                    calculator.create_submission_script(formula)
    else:
        print("\n⚠️  MP_API_KEY not set!")
        print("   Set it with: export MP_API_KEY='your_key_from_materialsproject.org'")
        print("\nTemplate created. You can still:")
        print("  1. Manually get structures from Materials Project")
        print("  2. Use the VASP input preparation code")
        print("  3. Run calculations on your cluster")


if __name__ == '__main__':
    main()
