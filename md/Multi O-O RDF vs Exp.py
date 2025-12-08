"""
O-O RDF at different temperatures and 1 g/mL computed via classical MD simulations in the NVT ensemble.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.io import Trajectory
from ase.geometry import get_distances
import time
#import pandas as pd
import os


def read_experimental_data(exp_file):

    try:
        data = []
        with open(exp_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    values = line.strip().split('\t')
                    if len(values) >= 6:
                        try:
                            r = float(values[3])  
                            g_oo = float(values[4])  
                            if r > 0 and not np.isnan(g_oo):
                                data.append([r, g_oo])
                        except ValueError:
                            continue
            
        data = np.array(data)
        r_exp = data[:, 0]
        g_oo_exp = data[:, 1]e
        
        return r_exp, g_oo_exp
        


def process_experimental_data_smooth(r_exp, g_oo_exp, start_cutoff=2.35, cutoff=2.5):
    r_processed = r_exp.copy()
    g_processed = g_oo_exp.copy()

    mask_zero = r_processed < start_cutoff
    mask_transition = (r_processed >= start_cutoff) & (r_processed <= cutoff)

    g_processed[mask_zero] = 0.0

    if np.any(mask_transition):
        r_transition = r_processed[mask_transition]
        g_transition_orig = g_oo_exp[mask_transition]

        width = (cutoff - start_cutoff) / 3.0

        smooth_factor = np.exp(-((cutoff - r_transition) / width) ** 2)

        g_processed[mask_transition] = g_transition_orig * smooth_factor

    
    return r_processed, g_processed


def calculate_oo_rdf(traj_file, rcut=6.0, nbins=300, skip_frames=0):

    try:
        traj = Trajectory(traj_file, 'r')

    
    total_frames = len(traj)
    print(f" {total_frames}")
    
    if skip_frames >= total_frames:
        return None, None

    dr = rcut / nbins
    r = np.linspace(dr/2, rcut - dr/2, nbins)

    rdf_hist = np.zeros(nbins)
    frame_count = 0
    
    start_time = time.time()

    for frame_idx, atoms in enumerate(traj):
        if frame_idx < skip_frames:
            continue

        oxygen_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) 
                         if symbol == 'O']
        
        if len(oxygen_indices) < 2:
            continue

        positions = atoms.get_positions()
        oxygen_positions = positions[oxygen_indices]

        cell = atoms.get_cell()

        if cell.any(): 
            distances = []
            for i in range(len(oxygen_indices)):
                for j in range(i+1, len(oxygen_indices)):
                
                    vec = oxygen_positions[j] - oxygen_positions[i]
                  
                    if np.linalg.norm(cell.diagonal()) > 0:
                        vec = vec - np.round(vec @ np.linalg.inv(cell)) @ cell
                    dist = np.linalg.norm(vec)
                    if dist < rcut:
                        distances.append(dist)
        else:  
            distances = []
            for i in range(len(oxygen_indices)):
                for j in range(i+1, len(oxygen_indices)):
                    dist = np.linalg.norm(oxygen_positions[j] - oxygen_positions[i])
                    if dist < rcut:
                        distances.append(dist)

        if distances:
            hist, _ = np.histogram(distances, bins=nbins, range=(0, rcut))
            rdf_hist += hist
        
        frame_count += 1

        if (frame_idx + 1) % 200 == 0:
            elapsed = time.time() - start_time
            print(f"    {frame_idx + 1}/{total_frames} : {elapsed:.1f}s)")
    

    last_atoms = traj[-1]
    oxygen_indices = [i for i, symbol in enumerate(last_atoms.get_chemical_symbols()) 
                     if symbol == 'O']
    n_oxygen = len(oxygen_indices)
    volume = last_atoms.get_volume()

    rho = n_oxygen / volume
    
    shell_volumes = 4 * np.pi * r**2 * dr
    expected_pairs = (n_oxygen * (n_oxygen - 1) / 2) * frame_count
    expected_counts = rho * shell_volumes * expected_pairs / n_oxygen

    mask = expected_counts > 0
    g_r = np.zeros_like(r)
    g_r[mask] = rdf_hist[mask] / expected_counts[mask]
    
    elapsed_total = time.time() - start_time
    print(f"用时: {elapsed_total:.1f}s")
    
    return r, g_r


def record_curve_values(rdf_data_dict, r_exp_processed=None, g_exp_processed=None):
    curve_records = {}

    sorted_temps = sorted(rdf_data_dict.keys())
    
    for temp in sorted_temps:
        r_md, g_r_md = rdf_data_dict[temp]
        if r_md is not None and g_r_md is not None:
            record = analyze_rdf_curve(r_md, g_r_md, f"{temp}K")
            curve_records[f"MD_{temp}K"] = record

    if r_exp_processed is not None and g_exp_processed is not None:
        record = analyze_rdf_curve(r_exp_processed, g_exp_processed, "EXP 295K")
        curve_records["EXP_295K"] = record
    
    print(f"{'='*80}")
    
    save_curve_records(curve_records)
    
    return curve_records


def analyze_rdf_curve(r, g_r, curve_name):

    
    record = {
        'name': curve_name,
        'peak_position': None,
        'peak_height': None,
        'first_min_position': None,
        'first_min_depth': None,
        'coordination_number': None
    }
    
    try:
        mask_peak = (r >= 2.5) & (r <= 3.5)
        if np.any(mask_peak):
            r_peak = r[mask_peak]
            g_peak = g_r[mask_peak]
            peak_idx = np.argmax(g_peak)
            record['peak_position'] = r_peak[peak_idx]
            record['peak_height'] = g_peak[peak_idx]

            dr = r[1] - r[0] if len(r) > 1 else 0.02
            rho_bulk = 0.033  
            coordination_number = 4 * np.pi * rho_bulk * np.trapz(g_peak * r_peak**2, dx=dr)
            record['coordination_number'] = coordination_number

        mask_min = (r >= 3.0) & (r <= 4.5)
        if np.any(mask_min):
            r_min = r[mask_min]
            g_min = g_r[mask_min]
            min_idx = np.argmin(g_min)
            record['first_min_position'] = r_min[min_idx]
            record['first_min_depth'] = g_min[min_idx]

        print(f"{curve_name:<15} {record['peak_position']:<12.3f} {record['peak_height']:<10.3f} "
              f"{record['first_min_position']:<15.3f} {record['first_min_depth']:<12.3f} "
              f"{record['coordination_number']:<18.2f}")
              
    except Exception as e:
        print(f"{curve_name:<15} {str(e)}")
    
    return record


def plot_multi_temperature_rdf(rdf_data_dict, r_exp_processed=None, g_exp_processed=None, 
                               save_data=True, save_plot=True):

    plt.figure(figsize=(12, 8))

    temperature_colors = {
        300: "#CD1212FF",   # 红色 - 低温
        500: "#7c7f18",   # 橙色 
        1000: "#21795d",  # 绿色 - 中温
        1500: "#1d53b6",  # 蓝色
        2000: "#710b4f"   # 紫色 - 高温
    }

    if r_exp_processed is not None and g_exp_processed is not None:

        mask = (r_exp_processed >= 2.5) & (r_exp_processed <= 6.0)
        plt.plot(r_exp_processed[mask], g_exp_processed[mask], 'k-', linewidth=3.0, 
                label='EXP 295 K', alpha=0.9, zorder=10)
    
    sorted_temps = sorted(rdf_data_dict.keys())
    
    for temp in sorted_temps:
        r_md, g_r_md = rdf_data_dict[temp]
        if r_md is not None and g_r_md is not None:
            color = temperature_colors.get(temp, '#666666')
            plt.plot(r_md, g_r_md, '-', color=color, linewidth=2.5, 
                    label=f'{temp} K', alpha=0.8)

    plt.xlabel('r [Å]', fontsize=18, fontweight='bold')
    plt.ylabel('g$_{OO}$(r)', fontsize=18, fontweight='bold')
    #plt.title('Oxygen-Oxygen Radial Distribution Function\nCAMP MD vs X-ray Diffraction (Processed)', fontsize=16, pad=20)

    plt.xlim(2, 6)
    plt.ylim(-0.2, 3)

    plt.legend(fontsize=14, frameon=True, fancybox=True, shadow=True,
               loc='upper right', ncol=1)

    plt.tick_params(axis='both', which='major', labelsize=14, width=1.5)
    plt.tick_params(axis='both', which='minor', labelsize=12, width=1)
    
    plt.tight_layout()

    if save_plot:
        plt.savefig('OO_RDF_Multi_Temperature_Processed.png', dpi=300, bbox_inches='tight')
        plt.savefig('OO_RDF_Multi_Temperature_Processed.pdf', bbox_inches='tight')
    
    plt.show()

    if save_data:
        for temp in sorted_temps:
            r_md, g_r_md = rdf_data_dict[temp]
            if r_md is not None and g_r_md is not None:
                data_md = np.column_stack([r_md, g_r_md])
                np.savetxt(f'OO_RDF_MD_{temp}K.dat', data_md, 
                           header=f'r(Å)  g_OO(r)  # CACE MD trajectory at {temp}K',
                           fmt='%.6f')

        if r_exp_processed is not None and g_exp_processed is not None:
            mask = (r_exp_processed >= 2.5) & (r_exp_processed <= 6.0)
            data_exp = np.column_stack([r_exp_processed[mask], g_exp_processed[mask]])
            np.savetxt('OO_RDF_Experiment_Processed.dat', data_exp, 
                       header='r(Å)  g_OO(r)  # X-ray diffraction experiment at 295K (processed)',
                       fmt='%.6f')

        save_combined_data(rdf_data_dict, r_exp_processed, g_exp_processed)


def save_combined_data(rdf_data_dict, r_exp_processed=None, g_exp_processed=None):
    
    r_min, r_max = 2.5, 6.0
    r_unified = np.linspace(r_min, r_max, 500)
    data_columns = [r_unified]
    headers = ['r(Å)']

    sorted_temps = sorted(rdf_data_dict.keys())
    for temp in sorted_temps:
        r_md, g_r_md = rdf_data_dict[temp]
        if r_md is not None and g_r_md is not None:
            g_interp = np.interp(r_unified, r_md, g_r_md)
            data_columns.append(g_interp)
            headers.append(f'g_OO_{temp}K')

    if r_exp_processed is not None and g_exp_processed is not None:
        mask = (r_exp_processed >= r_min) & (r_exp_processed <= r_max)
        g_exp_interp = np.interp(r_unified, r_exp_processed[mask], g_exp_processed[mask])
        data_columns.append(g_exp_interp)
        headers.append('g_OO_EXP_295K_Processed')

    combined_data = np.column_stack(data_columns)

    header_str = '  '.join(f'{h:>12}' for h in headers)
    header_str = f"# {header_str}\n# Multi-temperature O-O RDF comparison - CACE MD vs X-ray experiment (processed)"

    np.savetxt('OO_RDF_All_Temperatures_Processed.dat', combined_data, 
               header=header_str, fmt='%12.6f')


def main():

    temperatures = [300, 500, 1000, 1500, 2000]
    traj_files = {
        300: "water_md_data/md-T300_rdf.traj",
        500: "water_md_data/md-T500_rdf.traj", 
        1000: "water_md_data/md-T1000_rdf.traj",
        1500: "water_md_data/md-T1500_rdf.traj",
        2000: "water_md_data/md-T2000_rdf.traj"
    }
    
    exp_file = "Ambient_water_xray_data.txt"  

    rcut = 6.0      
    nbins = 300    
    skip_frames = 50  
    

    start_cutoff = 2.35  
    cutoff = 2.5         
    

    r_exp_original, g_oo_exp_original = read_experimental_data(exp_file)
    
    r_exp_processed, g_exp_processed = None, None
    if r_exp_original is not None and g_oo_exp_original is not None:
        r_exp_processed, g_exp_processed = process_experimental_data_smooth(
            r_exp_original, g_oo_exp_original, 
            start_cutoff=start_cutoff, cutoff=cutoff
        )

    rdf_data_dict = {}
    
    for temp in temperatures:
        traj_file = traj_files[temp]
        
        if os.path.exists(traj_file):
            r_md, g_r_md = calculate_oo_rdf(traj_file, rcut=rcut, nbins=nbins, skip_frames=skip_frames)
        
        else:
            print(f"{traj_file}")
    
    curve_records = record_curve_values(rdf_data_dict, r_exp_processed, g_exp_processed)
    
    
    plot_multi_temperature_rdf(rdf_data_dict, r_exp_processed, g_exp_processed, 
                               save_data=True, save_plot=True)


if __name__ == "__main__":
    main()