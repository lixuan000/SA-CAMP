import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, Trajectory
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import time
import os

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.unicode_minus'] = False

def load_experimental_rdf_data():

    exp_files = {
        'H-H': 'water_data/298K_case(b)_ghh.csv',
        'O-H': 'water_data/298K_case(b)_goh.csv', 
        'O-O': 'water_data/298K_case(b)_goo.csv'
    }
    
    rdf_data = {}
    
    for pair, filename in exp_files.items():
        if not os.path.exists(filename):
            print(f"warning{filename} ")
            continue
            
        print(f"{filename}")
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            r_values = []
            g_values = []

            for line in lines[3:]: 
                line = line.strip()
                if line and not line.startswith('Bin'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            r_val = float(parts[1]) 
                            g_val = float(parts[2])  
                            r_values.append(r_val)
                            g_values.append(g_val)
                        except ValueError:
                            continue
            
            if r_values and g_values:
                r_values = np.array(r_values)
                g_values = np.array(g_values)

                mask = r_values <= 6.0
                r_filtered = r_values[mask]
                g_filtered = g_values[mask]

                if len(g_filtered) > 10:
                    diff = np.abs(np.diff(g_filtered))
                    small_change_mask = diff < 1e-6
                    for i in range(len(small_change_mask)):
                        if small_change_mask[i] and r_filtered[i] > 1.0:
                            g_filtered[i+1] += np.random.normal(0, 0.001)
                
                rdf_data[pair] = {
                    'r': r_filtered,
                    'g_r': g_filtered
                }
                print(f"   {len(r_filtered)}")
                print(f" r = {r_filtered.min():.3f}-{r_filtered.max():.3f} Å, "
                      f"g(r) = {g_filtered.min():.3f}-{g_filtered.max():.3f}")
            else:
                
    
    return rdf_data

def calculate_rdf_fixed(trajectory_file, atom1, atom2, rcut=6.0, nbins=300, skip_frames=0):
    try:
        if trajectory_file.endswith('.traj'):
            traj = Trajectory(trajectory_file, 'r')
        else:
            frames = read(trajectory_file, ":")
            if not isinstance(frames, list):
                frames = [frames]
            traj = frames
    except:
        print(f"{trajectory_file}")
        return None, None
    
    total_frames = len(traj)
    print(f"{total_frames}")
    
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

        symbols = atoms.get_chemical_symbols()
        indices1 = [i for i, symbol in enumerate(symbols) if symbol == atom1]
        indices2 = [i for i, symbol in enumerate(symbols) if symbol == atom2]
        
        if len(indices1) == 0 or len(indices2) == 0:
            continue
        
        positions = atoms.get_positions()
        positions1 = positions[indices1]
        positions2 = positions[indices2]

        cell = atoms.get_cell()

        distances = []
        
        if atom1 == atom2:
            for i in range(len(indices1)):
                for j in range(i+1, len(indices1)):
                    vec = positions1[j] - positions1[i]

                    if cell.any():
                        if np.linalg.norm(cell.diagonal()) > 0:
                            vec = vec - np.round(vec @ np.linalg.inv(cell)) @ cell
                    
                    dist = np.linalg.norm(vec)
                    if dist < rcut:
                        distances.append(dist)
        else:
            for i in range(len(indices1)):
                for j in range(len(indices2)):
                    vec = positions2[j] - positions1[i]
                    
                    if cell.any():
                        if np.linalg.norm(cell.diagonal()) > 0:
                            vec = vec - np.round(vec @ np.linalg.inv(cell)) @ cell
                    
                    dist = np.linalg.norm(vec)
                    if dist < rcut:
                        distances.append(dist)

        if distances:
            hist, _ = np.histogram(distances, bins=nbins, range=(0, rcut))
            rdf_hist += hist
        
        frame_count += 1

        if (frame_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            progress = (frame_idx + 1 - skip_frames) / (total_frames - skip_frames) * 100
            print(f"{frame_idx + 1}/{total_frames}  ({progress:.1f}%) "
                  f"({elapsed:.1f})")
    
    if frame_count == 0:
        return None, None

    last_atoms = traj[-1]
    symbols = last_atoms.get_chemical_symbols()
    n1 = len([s for s in symbols if s == atom1])
    n2 = len([s for s in symbols if s == atom2])
    volume = last_atoms.get_volume()
    
    print(f"{atom1}: {n1}")
    print(f"{atom2}: {n2}")volume
    print(f"volume: {volume:.2f} Ų")

    if atom1 == atom2:
        rho = n1 / volume  
        expected_pairs = (n1 * (n1 - 1) / 2) * frame_count  
        shell_volumes = 4 * np.pi * r**2 * dr
        expected_counts = rho * shell_volumes * expected_pairs / n1  
    else:

        rho = n2 / volume  
        expected_pairs = n1 * n2 * frame_count 
        shell_volumes = 4 * np.pi * r**2 * dr
        expected_counts = rho * shell_volumes * expected_pairs / n2 

    mask = expected_counts > 0
    g_r = np.zeros_like(r)
    g_r[mask] = rdf_hist[mask] / expected_counts[mask]
    
    elapsed_total = time.time() - start_time
    
    return r, g_r

def calculate_rdf_pairs_with_options(trajectory_file, pairs=['O-O', 'O-H', 'H-H'], 
                                   r_max=6.0, n_bins=300, skip_frames=50, 
                                   fast_mode=True):

    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"{trajectory_file} ")

    if trajectory_file.endswith('.traj'):
        traj = Trajectory(trajectory_file, 'r')
    else:
        frames = read(trajectory_file, ":")
        if not isinstance(frames, list):
            frames = [frames]
        traj = frames
    
    total_frames = len(traj)
    
    if fast_mode:
        max_frames = min(1000, total_frames - skip_frames)
        step = max(1, (total_frames - skip_frames) // max_frames)
    else:
        step = 1
        max_frames = total_frames - skip_frames
    
    
    rdf_results = {}
    
    for pair in pairs:
        atom1, atom2 = pair.split('-')
        print(f"\n{'='*50}")
        
        if fast_mode and step > 1:
            r, g_r = calculate_rdf_with_step(trajectory_file, atom1, atom2, 
                                           rcut=r_max, nbins=n_bins, 
                                           skip_frames=skip_frames, step=step)
        else:
            r, g_r = calculate_rdf_fixed(trajectory_file, atom1, atom2, 
                                       rcut=r_max, nbins=n_bins, skip_frames=skip_frames)
        
        if r is not None and g_r is not None:
            rdf_results[pair] = {
                'r': r,
                'g_r': g_r
            }
        else:
    
    return rdf_results


def plot_individual_rdf_comparison(pair, rdf_md, rdf_exp, colors):
    plt.figure(figsize=(10, 8))
    
    color = colors[pair]

    if pair in rdf_md:
        r_md = rdf_md[pair]['r']
        g_md = rdf_md[pair]['g_r']
        g_md_smooth = gaussian_filter1d(g_md, sigma=1.0)
        plt.plot(r_md, g_md_smooth, color=color, linestyle='-', linewidth=2.5, 
                label=f'MD Simulation', alpha=0.9)

    if pair in rdf_exp:
        r_exp = rdf_exp[pair]['r']
        g_exp_smooth = gaussian_filter1d(g_exp, sigma=1.0)
        plt.plot(r_exp, g_exp_smooth, color=color, linestyle='--', linewidth=2.5,
                label=f'Experiment', alpha=0.8)
    
    plt.xlabel('r [Å]', fontsize=16, weight='bold')
    plt.ylabel(f'g$_{{{pair.replace("-", "")}}}$(r)', fontsize=16, weight='bold')
    #plt.title(f'{pair} Radial Distribution Function', fontsize=18, weight='bold', pad=20)
    plt.xlim(0, 6)
    plt.ylim(0, 3.5)
    
    #plt.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    #plt.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.legend(fontsize=14, loc='upper right', framealpha=1.0, 
              fancybox=False, edgecolor='black')

    plt.tick_params(axis='both', which='major', labelsize=14)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()

    output_name = f'{pair.replace("-", "")}_RDF_comparison_fixed'
    plt.savefig(f'{output_name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    if pair in rdf_md and pair in rdf_exp:
        g_md = gaussian_filter1d(rdf_md[pair]['g_r'], sigma=1.0)
        g_exp = gaussian_filter1d(rdf_exp[pair]['g_r'], sigma=1.0)

        if pair == 'O-H':
            peak_range = (0.8, 2.0)
        elif pair == 'H-H':
            peak_range = (1.2, 2.5)
        elif pair == 'O-O':
            peak_range = (2.0, 4.0)
        else:
            peak_range = (0.5, 4.0)
        
        r_md = rdf_md[pair]['r']
        r_exp = rdf_exp[pair]['r']

        mask_md = (r_md >= peak_range[0]) & (r_md <= peak_range[1])
        mask_exp = (r_exp >= peak_range[0]) & (r_exp <= peak_range[1])
        
        if np.any(mask_md) and np.any(mask_exp):
            peak_pos_md = r_md[mask_md][np.argmax(g_md[mask_md])]
            peak_pos_exp = r_exp[mask_exp][np.argmax(g_exp[mask_exp])]
            peak_height_md = np.max(g_md[mask_md])
            peak_height_exp = np.max(g_exp[mask_exp])


def calculate_mae(rdf_md, rdf_exp):
    mae_results = {}

    for pair in rdf_md:
        if pair in rdf_exp:
            r_md = rdf_md[pair]['r']
            g_md = rdf_md[pair]['g_r']
            r_exp = rdf_exp[pair]['r']
            g_exp = rdf_exp[pair]['g_r']

            r_common = r_md
            g_exp_interp = np.interp(r_common, r_exp, g_exp)

            mask = (r_common >= 0.5) & (r_common <= 6.0)
            r_calc = r_common[mask]
            g_md_calc = g_md[mask] 
            g_exp_calc = g_exp_interp[mask]

            mae = np.mean(np.abs(g_md_calc - g_exp_calc))
            mae_results[pair] = mae
            print(f"  {pair}: MAE = {mae:.4f} (: {r_calc.min():.1f}-{r_calc.max():.1f} Å)")
    
    return mae_results

if __name__ == "__main__":
    traj_file = "water_md_data/md-water-T300.traj"
    pairs = ['O-O', 'O-H', 'H-H']
    colors = {
        'O-O': 'red',
        'O-H': 'blue', 
        'H-H': 'green'
    }
    
    
    use_fast_mode = False
    
    
    rdf_exp = load_experimental_rdf_data()

    rdf_md = calculate_rdf_pairs_with_options(traj_file, pairs=pairs, r_max=6.0, 
                                            n_bins=300, skip_frames=50, 
                                            fast_mode=fast_mode)

    mae_results = calculate_mae(rdf_md, rdf_exp)
    
    total_mae = 0
    for pair, mae in mae_results.items():
        print(f"{pair}: {mae:.4f}")
        total_mae += mae
    print(f"average MAE: {total_mae/len(mae_results):.4f}")
    
    
    for pair in pairs:
        plot_individual_rdf_comparison(pair, rdf_md, rdf_exp, colors)
    