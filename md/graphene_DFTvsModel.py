"""
Interlayer binding energy curves as a function of interlayer distance displacement for AB and AA stacking configurations
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ase import Atoms
import os
import sys
import warnings
warnings.filterwarnings('ignore')


try:
    from camp.ase.calculator import CAMPCalculator
except ImportError:
    sys.exit(1)

plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

class IntegratedGrapheneBilayerAnalysis:
    
    def __init__(self, model_path):
        self.model_path = model_path
        
        
        print(f"{model_path}")
        self.calc = CAMPCalculator(model_path, device="cpu")
        self.a = 2.466
        self.d0 = 3.4   
    
    def create_graphene_sheet(self, size=(3, 3), vacuum=15.0):
        nx, ny = size
        
        unit_cell_positions = np.array([
            [0.0, 0.0, 0.0],
            [self.a / 2.0, self.a / (2.0 * np.sqrt(3)), 0.0]
        ])
        
        a1 = np.array([self.a, 0.0, 0.0])
        a2 = np.array([self.a / 2.0, self.a * np.sqrt(3) / 2.0, 0.0])
        
        positions = []
        for i in range(nx):
            for j in range(ny):
                for base_pos in unit_cell_positions:
                    pos = base_pos + i * a1 + j * a2
                    positions.append(pos)
        
        positions = np.array(positions)
        
        cell = np.array([
            [nx * a1[0], 0.0, 0.0],
            [ny * a2[0], ny * a2[1], 0.0],
            [0.0, 0.0, 2 * vacuum]
        ])
        
        positions[:, 2] += vacuum
        
        symbols = ['C'] * len(positions)
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        
        return atoms
    
    def create_isolated_layers(self, stacking='AB', size=(3, 3), separation=20.0):
        layer = self.create_graphene_sheet(size=size, vacuum=15.0)
        positions = layer.get_positions()
        cell = layer.get_cell()
        n_atoms = len(positions)
        

        cell[2, 2] = 2 * separation + 30.0
        

        bilayer_positions = np.zeros((2 * n_atoms, 3))
        
        bottom_positions = positions.copy()
        bottom_positions[:, 2] = separation / 2.0  
        bilayer_positions[:n_atoms] = bottom_positions
        
        top_positions = positions.copy()
        top_positions[:, 2] = cell[2, 2] - separation / 2.0 
        
        if stacking == 'AB':
            dx = self.a / 3.0
            dy = self.a / (3.0 * np.sqrt(3))
            top_positions[:, 0] += dx
            top_positions[:, 1] += dy
        elif stacking == 'AA':
            pass
        
        bilayer_positions[n_atoms:] = top_positions
        
        separated_bilayer = Atoms(
            symbols=['C'] * (2 * n_atoms),
            positions=bilayer_positions,
            cell=cell,
            pbc=True
        )
        
        separated_bilayer.set_calculator(self.calc)
        
        return separated_bilayer
    
    def create_bilayer_graphene(self, stacking='AB', interlayer_distance=3.4, size=(3, 3)):
        layer = self.create_graphene_sheet(size=size, vacuum=15.0)
        positions = layer.get_positions()
        cell = layer.get_cell()
        n_atoms = len(positions)
        
        bilayer_positions = np.zeros((2 * n_atoms, 3))
        
        bilayer_positions[:n_atoms] = positions
        
        top_positions = positions.copy()
        top_positions[:, 2] += interlayer_distance
        
        if stacking == 'AB':
            dx = self.a / 3.0
            dy = self.a / (3.0 * np.sqrt(3))
            top_positions[:, 0] += dx
            top_positions[:, 1] += dy
        elif stacking == 'AA':
            pass
        
        bilayer_positions[n_atoms:] = top_positions
        
        bilayer = Atoms(
            symbols=['C'] * (2 * n_atoms),
            positions=bilayer_positions,
            cell=cell,
            pbc=True
        )
        
        bilayer.set_calculator(self.calc)
        
        return bilayer
    
    def calculate_interlayer_energy_curve(self, distance_range=None, size=(3, 3), method='separated_reference'):
        if distance_range is None:
            distance_range = np.arange(2.4, 6.2, 0.1)
        
        print(f" {method}")
        print(f"{distance_range[0]:.1f} - {distance_range[-1]:.1f} Å")
        print(f"{size[0]}×{size[1]}")
        
        if method == 'separated_reference':
            sep_ab = self.create_isolated_layers(stacking='AB', size=size, separation=20.0)
            sep_aa = self.create_isolated_layers(stacking='AA', size=size, separation=20.0)
            E_ref_ab = sep_ab.get_potential_energy()
            E_ref_aa = sep_aa.get_potential_energy()
            print(f"{E_ref_ab:.6f} eV")
            print(f"{E_ref_aa:.6f} eV")
        
        n_atoms_per_layer = size[0] * size[1] * 2  
        n_total_atoms = 2 * n_atoms_per_layer 
        
        results = {
            'distances': distance_range,
            'AB': [],
            'AA': [],
            'method': method
        }
        
        for stacking in ['AB', 'AA']:
            print(f"\n{stacking}...")
            energies = []
            success_count = 0
            
            E_ref = E_ref_ab if stacking == 'AB' else E_ref_aa
            
            for i, d in enumerate(distance_range):
                try:
                    bilayer = self.create_bilayer_graphene(
                        stacking=stacking,
                        interlayer_distance=d,
                        size=size
                    )
                    
        
                    E_bilayer = bilayer.get_potential_energy()
                    
                    
                    if method == 'separated_reference':
                        interlayer_energy = (E_bilayer - E_ref) * 1000.0 / n_total_atoms
                    else:
                        interlayer_energy = (E_bilayer - E_ref) * 1000.0 / n_total_atoms
                    
                    energies.append(interlayer_energy)
                    success_count += 1
                    
                    print(f"  d={d:.1f}Å: {interlayer_energy:.3f} meV/atom")
                    
            
            results[stacking] = np.array(energies)
            print(f"  {stacking}: {success_count}/{len(distance_range)} ")
        
        return results
    
    def read_csv_data(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            data_start = 0
            for i, line in enumerate(lines):
                if 'Distance_Å' in line:
                    data_start = i + 1
                    break

            data = []
            for i in range(data_start, len(lines)):
                line = lines[i].strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        try:
                            distance = float(parts[0])
                            delta_d = float(parts[1])
                            dft_energy = float(parts[2])
                            camp_energy = float(parts[3])
                            
                            data.append({
                                'distance': distance,
                                'delta_d': delta_d,
                                'dft_energy': dft_energy,
                                'camp_energy': camp_energy
                            })
                        except ValueError:
                            continue
            
            return pd.DataFrame(data)
            
    
    def plot_integrated_curves(self, camp_results, ab_csv_path=None, aa_csv_path=None, save_path="integrated_interlayer_energy.png"):
        fig, ax = plt.subplots(figsize=(8, 7))
        
        distances = camp_results['distances']
        method = camp_results.get('method', 'unknown')

        ab_energies = camp_results['AB']
        valid_ab = ab_energies[~np.isnan(ab_energies)]
        if len(valid_ab) > 0:
            ab_min_idx = np.nanargmin(ab_energies)
            d_ab_min = distances[ab_min_idx]
            E_ab_min = ab_energies[ab_min_idx]
        else:
            d_ab_min = self.d0
            E_ab_min = 0

        delta_d = distances - d_ab_min
        
        ab_energies_plot = camp_results['AB'] - E_ab_min
        ab_mask = ~np.isnan(ab_energies_plot)
        
        if np.sum(ab_mask) > 0:
            ax.plot(delta_d[ab_mask], ab_energies_plot[ab_mask], 
                    '-', color='#2E8B57', linewidth=3,
                    label='SA-CAMP AB', alpha=0.9)
        
        aa_energies_plot = camp_results['AA'] - E_ab_min
        aa_mask = ~np.isnan(aa_energies_plot)
        
        if np.sum(aa_mask) > 0:
            ax.plot(delta_d[aa_mask], aa_energies_plot[aa_mask], 
                    '-', color='#DC143C', linewidth=3,
                    label='SA-CAMP AA', alpha=0.9)
        
        if ab_csv_path:
            ab_dft_data = self.read_csv_data(ab_csv_path)
            if ab_dft_data is not None and not ab_dft_data.empty:
                ax.scatter(ab_dft_data['delta_d'], ab_dft_data['dft_energy'], 
                          color='#2E8B57', marker='o', s=100, alpha=0.8, 
                          edgecolors='white', linewidth=1.5, 
                          label='DFT AB', zorder=5)
                print(f"   DFT AB: {len(ab_dft_data)} 个")
        
        if aa_csv_path:
            aa_dft_data = self.read_csv_data(aa_csv_path)
            if aa_dft_data is not None and not aa_dft_data.empty:
                ax.scatter(aa_dft_data['delta_d'], aa_dft_data['dft_energy'], 
                          color='#DC143C', marker='s', s=100, alpha=0.8, 
                          edgecolors='white', linewidth=1.5, 
                          label='DFT AA', zorder=6)
                print(f"   DFT AA: {len(aa_dft_data)} 个")

        ax.set_xlim(-1, 2.5)
        ax.set_ylim(-5, 60)
        
        ax.set_xticks(np.arange(-1, 3, 1))
        ax.set_yticks([0, 20, 40, 60])

        ax.set_xlabel('Δd = d - d₀ (Å)', fontsize=16, fontweight='bold')
        ax.set_ylabel('ΔE (meV/atom)', fontsize=16, fontweight='bold')

        legend = ax.legend(loc='upper right', fontsize=12, frameon=True, 
                          fancybox=True, shadow=True, framealpha=0.95)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')

        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)

        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png', transparent=False)
        plt.show()

def main():
    model_paths = [
        "graphene_model/epoch=972-step=601314.ckpt"
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    try:
        analyzer = IntegratedGrapheneBilayerAnalysis(model_path)
        
        
        distance_range = np.arange(2.4, 6.2, 0.1)
        supercell_size = (3, 3)
        
        camp_results = analyzer.calculate_interlayer_energy_curve(
            distance_range=distance_range,
            size=supercell_size,
            method='separated_reference'
        )

        ab_csv = 'graphene_data/ab_points.csv'
        aa_csv = 'graphene_data/aa_points.csv'
        
        
        analyzer.plot_integrated_curves(
            camp_results=camp_results,
            ab_csv_path=ab_csv,
            aa_csv_path=aa_csv,
            save_path="integrated_interlayer_energy.png"
        )
        

if __name__ == "__main__":
    main()