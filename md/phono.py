# carbon-diamond phono spectrum 
import torch
import yaml
import numpy as np
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list 
import hotphono
from phonopy import Phonopy
from phonopy.interface.phonopy_yaml import PhonopyYaml
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path

from camp.model.camp_transformer import CAMPTransformer
from camp.training.lit_model import LitModel
from camp.data.transform import ConsecutiveAtomType
from camp.data.data import Config

class SACAMPCalculator(Calculator):

    implemented_properties = [
        "energy", "energies", "forces", "stress", 
        "dipole", "polarizability", "descriptor"
    ]
    
    def __init__(self, model_path, config_path=None, energy_scale=1.0, energy_shift=0.0, 
                 device="cpu", double=True, **kwargs):
        Calculator.__init__(self, **kwargs)
        
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.precision = torch.float32
        if double:
            self.precision = torch.double
        
        self.model = self.load_model(model_path, config_path)
        self.model.eval()
        

        self.energy_scale = energy_scale
        self.energy_shift = energy_shift
        
 
        self.transform = ConsecutiveAtomType([6])  # Carbon
        
 
        self.cutoff = 6.0  
        
    def load_model(self, model_path, config_path):
        print(f"Loading SA-CAMP model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'hyper_parameters' in checkpoint and 'other_hparams' in checkpoint['hyper_parameters']:
            model_params = checkpoint['hyper_parameters']['other_hparams']['model'].copy()
            
            model_params['num_atom_types'] = int(model_params.get('num_atom_types', 1))
            model_params['r_cut'] = float(model_params.get('r_cut', 6.0))
            self.cutoff = model_params['r_cut'] 
            
            numeric_params = ['max_u', 'max_v', 'num_layers', 'max_chebyshev_degree', 
                            'hidden_dim', 'num_heads', 'dropout', 'ffn_dim', 'use_transformer_after']
            
            for param in numeric_params:
                if param in model_params:
                    if isinstance(model_params[param], str):
                        try:
                            if param == 'dropout':
                                model_params[param] = float(model_params[param])
                            else:
                                model_params[param] = int(float(model_params[param]))
                        except:
                            if param == 'dropout':
                                model_params[param] = 0.1
                            elif param == 'use_transformer_after':
                                model_params[param] = 0
                            else:
                                model_params[param] = 48

            list_params = ['radial_mlp_hidden_layers', 'output_mlp_hidden_layers']
            for param in list_params:
                if param in model_params and isinstance(model_params[param], str):
                    try:
                        model_params[param] = eval(model_params[param])
                    except:
                        model_params[param] = [48, 48]
                        
            model_params['atomic_energy_shift'] = None
            model_params['atomic_energy_scale'] = None
            
        else:
            model_params = {
                'num_atom_types': 1, 'max_u': 48, 'max_v': 3, 'num_average_neigh': 50.0,
                'num_layers': 3, 'r_cut': 6.0, 'max_chebyshev_degree': 8,
                'radial_mlp_hidden_layers': [48, 48], 'hidden_dim': 128, 'num_heads': 4,
                'dropout': 0.1, 'ffn_dim': 256, 'use_transformer_after': 0,
                'output_mlp_hidden_layers': [48, 48], 'atomic_energy_shift': None,
                'atomic_energy_scale': None,
            }
        
        try:
            model = CAMPTransformer(**model_params)

            state_dict = checkpoint['state_dict']
            model_state_dict = {}
            
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    clean_key = key[6:]
                    if clean_key in model.state_dict():
                        model_state_dict[clean_key] = value
            
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {len(unexpected_keys)} keys")

            model = model.to(self.device)

            if self.precision == torch.double:
                model = model.double()
            
            wrapper = ModelWrapper(model)
            return wrapper
    
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        
        try:
            config_data = Config.from_ase(atoms, r_cut=self.cutoff)
            config_data = self.transform(config_data)
            config_data = config_data.to(self.device)

            config_data.pos = config_data.pos.to(dtype=self.precision)
            if hasattr(config_data, 'cell') and config_data.cell is not None:
                config_data.cell = config_data.cell.to(dtype=self.precision)
            if hasattr(config_data, 'shift_vec') and config_data.shift_vec is not None:
                config_data.shift_vec = config_data.shift_vec.to(dtype=self.precision)

            if hasattr(config_data, 'edge_index'):
                config_data.edge_index = config_data.edge_index.long()
            if hasattr(config_data, 'atom_type'):
                config_data.atom_type = config_data.atom_type.long()
            if hasattr(config_data, 'num_atoms'):
                config_data.num_atoms = config_data.num_atoms.long()
            
            config_data.pos.requires_grad_(True)
            
            with torch.enable_grad():
                edge_vector = self._get_edge_vector(config_data)
                
                energy = self.model.model(
                    edge_vector=edge_vector,
                    edge_idx=config_data.edge_index,
                    atom_type=config_data.atom_type,
                    num_atoms=config_data.num_atoms
                )
                
                energy_total = energy.sum()
                energy_total = energy_total * self.energy_scale + self.energy_shift * len(atoms)
                
                forces = -torch.autograd.grad(
                    energy_total, config_data.pos, 
                    create_graph=False, retain_graph=False
                )[0] * self.energy_scale
            
            if "energy" in properties:
                self.results["energy"] = energy_total.detach().cpu().numpy().item()
            if "energies" in properties:
                self.results["energies"] = energy.detach().cpu().numpy()
            if "forces" in properties:
                self.results["forces"] = forces.detach().cpu().numpy()
            if "stress" in properties:
                self.results["stress"] = np.zeros(6)
    
    def _get_edge_vector(self, config_data):
        from camp.data.utils import get_edge_vec
        
        try:
            cell = config_data.cell
            shift_vec = config_data.shift_vec
        except AttributeError:
            cell = None
            shift_vec = None

        batch = torch.zeros(config_data.pos.shape[0], dtype=torch.long, device=config_data.pos.device)

        pos = config_data.pos

        if cell is not None:
            cell = cell.to(dtype=self.precision)
        
        if shift_vec is not None:
            shift_vec = shift_vec.to(dtype=self.precision)
        
        edge_vector = get_edge_vec(pos, shift_vec, cell, config_data.edge_index, batch)
        return edge_vector


def get_force_constants_sacamp(calc, phonon, displacement=0.01):
    print(f"{displacement} Å ")
    
    phonon.generate_displacements(distance=displacement)

    if hasattr(phonon, 'supercells_with_displacements'):
        supercells = phonon.supercells_with_displacements
    else:
        supercells = phonon.get_supercells_with_displacements()
    

    set_of_forces = []
    for i, supercell in enumerate(supercells):
        print(f"   {i+1}/{len(supercells)} ")
        
        from ase import Atoms
        atoms = Atoms(
            symbols=supercell.symbols,
            positions=supercell.positions,
            cell=supercell.cell,
            pbc=True
        )
        
        atoms.calc = calc
        forces = atoms.get_forces()

        forces -= np.mean(forces, axis=0)
        set_of_forces.append(forces)

    set_of_forces = np.array(set_of_forces)
    phonon.produce_force_constants(forces=set_of_forces)
    return phonon.force_constants


def plot_band_structure_improved(band_dict, axs, color="g", linestyle="-", max_freq=50, label=None):
    labels_path = band_dict["labels_path"]
    frequencies = band_dict["frequencies"]
    distances = band_dict["distances"]

    max_dist = distances[-1][-1]
    xscale = max_freq / max_dist * 1.5
    distances_scaled = [d * xscale for d in distances]

    n = 0
    if len(axs) > 0:
        axs[0].set_ylabel("Frequency(THz)", fontsize=14)
    
    legend_added = False
    
    for i, path in enumerate(labels_path):
        if i >= len(axs):
            break
            
        for spine in axs[i].spines.values():
            spine.set_linewidth(1.5)
        axs[i].tick_params(labelsize=14)
        
        xticks = [distances_scaled[n][0]]
        
        for j, label_name in enumerate(path[:-1]):
            xticks.append(distances_scaled[n][-1])
            
            axs[i].plot(
                [distances_scaled[n][-1], distances_scaled[n][-1]],
                [0, max_freq],
                linewidth=2, linestyle=":", c="grey",
            )

            plot_label = label if (i == 0 and j == 0 and not legend_added) else None
            axs[i].plot(
                distances_scaled[n], frequencies[n],
                linewidth=2, linestyle=linestyle, c=color, label=plot_label
            )

            if plot_label:
                legend_added = True
            
            n += 1
        
        axs[i].set_xlim(xticks[0], xticks[-1])
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels(path)
        axs[i].plot([xticks[0], xticks[-1]], [0, 0], linewidth=1, c="black")
        axs[i].set_ylim(0, max_freq)
    
    return axs

legend_handles = []
legend_labels = []

def add_to_legend(line, label):
    global legend_handles, legend_labels
    if label and label not in legend_labels:
        legend_handles.append(line)
        legend_labels.append(label)

def plot_legend_on_first_subplot(axs):
    global legend_handles, legend_labels
    if legend_handles and len(axs) > 0:
        axs[0].legend(legend_handles, legend_labels, 
                     loc='lower right', 
                     frameon=True, 
                     fancybox=True, 
                     shadow=True,
                     fontsize=12)


class ModelWrapper:
    def __init__(self, model):
        self.model = model
        
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def eval(self):
        self.model.eval()
        return self


def main():
    atoms = bulk('C', 'diamond', cubic=True)
    
    model_path = "E:/python_code/camp/gap17_model/9.24_gap17_v7.ckpt"
    config_path = "E:/python_code/camp/camp_run/models/GAP17/config_camp_transformer.yaml" if Path("config_camp_transformer.yaml").exists() else None
    energy_scale = 1.80
    energy_shift = 0.0
    
    try:
        calc = SACAMPCalculator(
            model_path, config_path, 
            energy_scale=energy_scale, 
            energy_shift=energy_shift,
            device="cuda" if torch.cuda.is_available() else "cpu",
            double=True 
        )
        atoms.calc = calc
        
        energy = atoms.get_potential_energy()
        print(f"energy: {energy}")
        
        forces = atoms.get_forces()
        print(f"forces shape: {forces.shape}")
        print(f"forces max: {np.max(np.abs(forces))}")
        
    except Exception as e:
        
        try:
            calc = SACAMPCalculator(
                model_path, config_path, 
                energy_scale=energy_scale, 
                energy_shift=energy_shift,
                device="cuda" if torch.cuda.is_available() else "cpu",
                double=False
            )
            atoms.calc = calc
            
            energy = atoms.get_potential_energy()
            print(f"energy: {energy}")
            
            forces = atoms.get_forces()
            print(f"forces shape: {forces.shape}")
            print(f"forces max: {np.max(np.abs(forces))}")
            
    
    
    try:
        unitcell = hotphono.ase2phono(atoms)
        supercell_matrix = np.array([2, 2, 2])
        phonon = Phonopy(unitcell=unitcell, 
                         supercell_matrix=supercell_matrix,
                         primitive_matrix='auto')
        
        fig = plt.figure(figsize=(15, 8))

        force_constants_dft = np.load("E:/python_code/camp/carbon_phono/force_constant_dft.npy")
        phonon.force_constants = force_constants_dft
        phpy_yaml = PhonopyYaml(settings={'force_sets': True, 'displacements': True,
                                          'force_constants': True, 'born_effective_charge': True,
                                          'dielectric_constant': True})
        phpy_yaml.set_phonon_info(phonon)
        atoms.info['phono_info'] = phpy_yaml._data
        band_dict_dft = hotphono.get_band_structure(phonon, atoms)

        n_paths = len(band_dict_dft['labels_path'])
        axs = ImageGrid(fig, 111, nrows_ncols=(1, n_paths), axes_pad=0.2, label_mode="L")

        global legend_handles, legend_labels
        legend_handles = []
        legend_labels = []

        print("绘制DFT参考数据...")
        plot_band_structure_improved(band_dict_dft, axs, 'black', '-', 50, 'DFT')
        line_dft, = axs[0].plot([], [], color='black', linestyle='-', linewidth=2, label='DFT')
        add_to_legend(line_dft, 'DFT')
        
        try:
            displacement = 0.01
            force_constants_sacamp = get_force_constants_sacamp(calc, phonon, displacement)
            phonon.force_constants = force_constants_sacamp
            phpy_yaml.set_phonon_info(phonon)
            atoms.info['phono_info'] = phpy_yaml._data
            band_dict_sacamp = hotphono.get_band_structure(phonon, atoms)

            plot_band_structure_improved(band_dict_sacamp, axs, 'red', '--', 50, 'SA-CAMP')

            line_sacamp, = axs[0].plot([], [], color='red', linestyle='--', linewidth=2, label='SA-CAMP')
            add_to_legend(line_sacamp, 'SA-CAMP')

            plot_legend_on_first_subplot(axs)

            dft_freqs = np.concatenate(band_dict_dft['frequencies'])
            sacamp_freqs = np.concatenate(band_dict_sacamp['frequencies'])
            
            if dft_freqs.shape == sacamp_freqs.shape:
                mae = np.mean(np.abs(sacamp_freqs - dft_freqs))
                rmse = np.sqrt(np.mean((sacamp_freqs - dft_freqs)**2))
                max_error = np.max(np.abs(sacamp_freqs - dft_freqs))
                
                print(f"  (MAE): {mae:.3f} THz")
                print(f"  (RMSE): {rmse:.3f} THz")
                print(f" {max_error:.3f} THz")
                
        
        plt.tight_layout()
        plt.savefig('GAP20-diamond_phonon_hotpp_style.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('GAP20-diamond_phonon_hotpp_style.png', dpi=300, bbox_inches='tight')
        
        plt.show()

if __name__ == "__main__":
    main()