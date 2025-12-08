#!/usr/bin/env python3
"""
water difussion
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.io import Trajectory
from scipy import stats
import os
from pathlib import Path

def unwrap_positions(positions, cell):

    n_frames, n_atoms, _ = positions.shape
    unwrapped = np.copy(positions)
    
    for t in range(1, n_frames):
        displacement = positions[t] - positions[t-1]

        for dim in range(3):
            large_positive = displacement[:, dim] > cell[dim] / 2
            large_negative = displacement[:, dim] < -cell[dim] / 2
            displacement[large_positive, dim] -= cell[dim]
            displacement[large_negative, dim] += cell[dim]
        unwrapped[t] = unwrapped[t-1] + displacement
    
    return unwrapped

def calculate_msd(positions, dt, cell=None):

    n_frames, n_atoms, _ = positions.shape
    if cell is not None:
        positions = unwrap_positions(positions, cell)
    msd = np.zeros(n_frames)
    time = np.arange(n_frames) * dt

    initial_positions = positions[0]
    
    for t in range(n_frames):
        displacement = positions[t] - initial_positions
        
        squared_displacement = np.sum(displacement**2, axis=1)
        msd[t] = np.mean(squared_displacement)
    
    return time, msd

def calculate_diffusion_coefficient(time, msd, fit_start_fraction=0.4, fit_end_fraction=0.9):
    start_idx = int(len(time) * fit_start_fraction)
    end_idx = int(len(time) * fit_end_fraction)
    
    if end_idx <= start_idx:
        end_idx = len(time)
    
    # MSD = 6Dt 
    #  D = slope / 6
    fit_time = time[start_idx:end_idx]
    fit_msd = msd[start_idx:end_idx]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(fit_time, fit_msd)
    
    #  (Å²/ps)
    D = slope / 6.0
    
    fit_range = (fit_time[0], fit_time[-1])
    
    return D, slope, r_value**2, fit_range

def extract_oxygen_positions(trajectory_file):

    traj = Trajectory(trajectory_file, 'r')

    first_frame = traj[0]
    oxygen_indices = [i for i, atom in enumerate(first_frame) if atom.symbol == 'O']
    
    print(f"Found {len(oxygen_indices)} oxygen atoms")
    print(f"Total atoms in system: {len(first_frame)}")

    expected_oxygens = 512
    if len(oxygen_indices) != expected_oxygens:
置
    positions = []
    cell_vectors = []
    
    for i, atoms in enumerate(traj):
        oxygen_pos = atoms.positions[oxygen_indices]
        positions.append(oxygen_pos)

        if atoms.cell is not None:
            cell_diag = np.diag(atoms.cell)
            cell_vectors.append(cell_diag)

            if i == 0:
                print(f"Cell parameters: {cell_diag[0]:.3f} x {cell_diag[1]:.3f} x {cell_diag[2]:.3f} Å")
    
    positions = np.array(positions)
    print(f"Trajectory length: {len(positions)} frames")

    if cell_vectors:
        cell = np.mean(cell_vectors, axis=0)
        expected_cell = 24.830476  # 12.415238 * 2
        if abs(cell[0] - expected_cell) > 1.0:
    else:
        cell = np.array([24.830476, 24.830476, 24.830476])
    
    timestep = 0.01
    
    return positions, timestep, cell

def validate_trajectory_data(positions, cell, temperature):

    n_frames, n_atoms, _ = positions.shape
    
    print(f" {n_frames}")
    print(f" {n_atoms}")
    print(f" {n_atoms} ")
 
    pos_min = np.min(positions)
    pos_max = np.max(positions)
    print(f"  {pos_min:.2f} - {pos_max:.2f} Å")

    max_displacement = 0
    for i in range(1, min(10, n_frames)): 
        displacement = np.linalg.norm(positions[i] - positions[i-1], axis=1)
        max_disp_frame = np.max(displacement)
        max_displacement = max(max_displacement, max_disp_frame)
    
    print(f" {max_displacement:.3f} Å")

    if temperature <= 500:
        reasonable_max_disp = 0.6
    elif temperature <= 1000:
        reasonable_max_disp = 0.8  
    else:
        reasonable_max_disp = 1.0  
    if max_displacement > reasonable_max_disp:
        print(f"   {reasonable_max_disp:.3f} Å）")析
    
    return True

def plot_msd_comparison(temperatures, time_data, msd_data, diffusion_coefficients, fit_data):

    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    temperature_colors = {
        300: "#CD1212FF",   # 红色 - 低温
        500: "#7c7f18",     # 橙色 
        1000: "#21795d",    # 绿色 - 中温
        1500: "#1d53b6",    # 蓝色
        2000: "#710b4f"     # 紫色 - 高温
    }
    
    legend_labels = []

    for i, (temp, time, msd, D, fit_info) in enumerate(zip(temperatures, time_data, msd_data, diffusion_coefficients, fit_data)):
        color = temperature_colors.get(temp, "#000000")  

        D_ang2_ps = D 

        ax.plot(time, msd, color=color, linewidth=2.5, alpha=0.9)
 
        legend_labels.append(f'{temp} K, D = {D_ang2_ps:.2f} Å²/ps')
    

    ax.set_xlabel('Time (ps)', fontsize=16, fontweight='bold')
    ax.set_ylabel('MSD (Å²)', fontsize=16, fontweight='bold')

    ax.set_xlim(-20, 320)  

    max_msd = max([np.max(msd) for msd in msd_data])
    ax.set_ylim(-max_msd*0.05, max_msd*1.05) 

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='major', width=1.5, length=6)
 
    legend = ax.legend(legend_labels, fontsize=13, frameon=True, 
                      fancybox=True, shadow=True, loc='upper left',
                      framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1)
    
    #ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    plt.tight_layout(pad=1.5)

    plt.savefig('cace_fig3b_reproduction_angstrom_units.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('cace_fig3b_reproduction_angstrom_units.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"cace_fig3b_reproduction_angstrom_units.png  cace_fig3b_reproduction_angstrom_units.pdf")
    plt.show()
    
    return fig, ax

def main():

    temperatures = [300, 500, 1000, 1500, 2000]  # K

    traj_files = [
        "water_md_data/md-T300_rdf5.traj",   # 300K
        "water_md_data/md-T500_rdf.traj",   # 500K
        "water_md_data/md-T1000_rdf.traj",  # 1000K
        "water_md_data/md-T1500_rdf.traj",  # 1500K
        "water_md_data/md-T2000_rdf.traj"   # 2000K
    ]
    

    time_data = []
    msd_data = []
    diffusion_coefficients = []
    fit_data = []
    valid_temperatures = []
    
    for temp, traj_file in zip(temperatures, traj_files):
        if not os.path.exists(traj_file):
            continue
        
        try:
            positions, dt, cell = extract_oxygen_positions(traj_file)
            print(f"step: {dt} ps")step
            if cell is not None:
                print(f"[{cell[0]:.3f}, {cell[1]:.3f}, {cell[2]:.3f}] Å")

            if not validate_trajectory_data(positions, cell, temp):
                continue

            time, msd = calculate_msd(positions, dt, cell)
            print(f"{time[-1]:.1f} ps")
            print(f"MSD: {msd[-1]:.1f} Å²")

            D, slope, r2, fit_range = calculate_diffusion_coefficient(time, msd)


            if r2 < 0.95:
                print(f"(R² = {r2:.4f})")

            time_data.append(time)
            msd_data.append(msd)
            diffusion_coefficients.append(D)
            fit_data.append((slope, r2, fit_range))
            valid_temperatures.append(temp)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue

    if time_data:
        print(f" {len(time_data)}")
        
        plot_msd_comparison(valid_temperatures, time_data, msd_data, 
                          diffusion_coefficients, fit_data)


        for temp, D, (slope, r2, fit_range) in zip(valid_temperatures, diffusion_coefficients, fit_data):
            print(f"{temp:<8} {D:<12.3f} {r2:<8.4f}")
        

        if valid_temperatures and valid_temperatures[0] == 300:
            calculated_D = diffusion_coefficients[0]  # Å²/ps
            print(f" {calculated_D:.3f} Å²/ps")
            experimental_D_angstrom = 0.267  # 2.67 × 10⁻⁵ cm²/s  Å²/ps
            relative_error = abs(calculated_D - experimental_D_angstrom)/experimental_D_angstrom*100
            print(f"{relative_error:.1f}%")
    else:
        print("")

if __name__ == "__main__":
    main()