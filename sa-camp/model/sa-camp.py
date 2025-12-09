"""CAMP-Transformer hybrid model class."""

from typing import Union, Optional

import torch
from torch import Tensor, nn

from camp.nn.embedding import AtomEmbedding
from camp.nn.linear import LinearMap

from .atomic_moment import AtomicMoment
from .hyper_moment import HyperMoment
from .readout import TotalEnergy
from .transformer import TransformerLayer
from .utils_jit import JITInterface


class CAMPTransformer(nn.Module):
    """Cartesian Atomic Moment Potential with Transformer Enhancement."""

    def __init__(
        self,
        num_atom_types: int,
        max_u: int,
        max_v: int,
        num_average_neigh: float,
        num_layers: int = 2,
        r_cut: float = 5.0,
        # radial
        max_chebyshev_degree: int = 8,
        radial_mlp_hidden_layers: Union[list[int], int] = 2,
        # transformer parameters
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_dim: int = 256,
        use_transformer_after: int = 0,  # layer index to start using transformer (0 = all layers)
        # output module
        output_mlp_hidden_layers: Union[list[int], int] = 2,
        atomic_energy_shift: Optional[Tensor] = None,
        atomic_energy_scale: Optional[Tensor] = None,
    ):
        """
        Args:
            num_atom_types: Number of atom types in the dataset
            max_u: Maximum radial degree
            max_v: Maximum angular degree
            num_average_neigh: Average number of neighbors of the atoms
            num_layers: Number of message passing layers
            r_cut: Cutoff radius for neighbor finding
            max_chebyshev_degree: Maximum degree of Chebyshev polynomial for radial basis
            radial_mlp_hidden_layers: Hidden layers for radial MLP
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            dropout: Dropout rate
            ffn_dim: Feed-forward network dimension in transformer
            use_transformer_after: Start using transformer after this layer index (0 means all layers)
            output_mlp_hidden_layers: Hidden layers for output MLP
            atomic_energy_shift: Shift for atomic energy
            atomic_energy_scale: Scale for atomic energy
        """
        super().__init__()
        self.num_atom_types = num_atom_types
        self.max_u = max_u
        self.max_v = max_v
        self.num_average_neigh = num_average_neigh
        self.num_layers = num_layers
        self.r_cut = r_cut
        self.hidden_dim = hidden_dim
        self.use_transformer_after = use_transformer_after

        self.max_chebyshev_degree = max_chebyshev_degree
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers

        self.output_mlp_hidden_layers = output_mlp_hidden_layers
        self.atomic_energy_shift = atomic_energy_shift
        self.atomic_energy_scale = atomic_energy_scale

        # Initial atom embedding
        self.atom_embedding = AtomEmbedding(num_atom_types, max_u + 1)

        # Feature dimension
        n_u = max_u + 1

        # Main processing layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # First layer uses only scalar features
            if i == 0:
                max_atom_feats_rank = 0
                mix_atom_feats_radial_channel = False
            else:
                max_atom_feats_rank = None
                mix_atom_feats_radial_channel = True

            # Last layer only produces scalar outputs for energy prediction
            if i == num_layers - 1:
                max_out_rank = 0
            else:
                max_out_rank = None

            # Create CAMP layer
            camp_layer = CAMPTransformerLayer(
                num_atom_types=num_atom_types,
                max_u=max_u,
                max_v=max_v,
                num_average_neigh=num_average_neigh,
                max_chebyshev_degree=max_chebyshev_degree,
                r_cut=r_cut,
                radial_mlp_hidden_layers=radial_mlp_hidden_layers,
                mix_atom_feats_radial_channel=mix_atom_feats_radial_channel,
                max_atom_feats_rank=max_atom_feats_rank,
                max_out_rank=max_out_rank,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                ffn_dim=ffn_dim,
                use_transformer=(i >= use_transformer_after)
            )
            
            self.layers.append(camp_layer)

        # Readout module for energy prediction
        self.readout = TotalEnergy(
            num_layers,
            max_u + 1,
            output_mlp_hidden_layers,
            atomic_energy_shift,
            atomic_energy_scale,
        )

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        num_atoms: Tensor,
    ) -> Tensor:
        """
        Forward pass of the CAMP-Transformer model.
        
        Args:
            edge_vector: Edge vectors between atoms
            edge_idx: Edge indices [2, num_edges]
            atom_type: Atom types [num_atoms]
            num_atoms: Number of atoms in each configuration [batch_size]
            
        Returns:
            Energy prediction tensor [batch_size]
        """
        # Initial atom embedding [n_atoms, embedding_dim]
        embedding = self.atom_embedding(atom_type)

        # Initial features - only scalars (v=0) for first layer
        atom_feats = {0: embedding.T}  # {v: (n_u, n_atoms)}

        # Store scalar features for readout
        scalar_feats = []
        
        # Process through layers
        for layer in self.layers:
            # Pass through CAMP-Transformer layer
            atom_feats = layer(edge_vector, edge_idx, atom_type, atom_feats)
            
            # Keep track of scalar features for energy prediction
            scalar_feats.append(atom_feats[0])

        # Predict energy from scalar features
        energy = self.readout(scalar_feats, atom_type, num_atoms)

        return energy


class CAMPTransformerLayer(nn.Module):
    """A CAMP layer enhanced with transformer for global message passing."""

    def __init__(
        self,
        num_atom_types: int,
        max_u: int,
        max_v: int,
        num_average_neigh: float,
        max_chebyshev_degree: int = 8,
        r_cut: float = 5.0,
        radial_mlp_hidden_layers: Union[list[int], int] = 2,
        mix_atom_feats_radial_channel: bool = True,
        max_atom_feats_rank: Optional[int] = None,
        max_out_rank: Optional[int] = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_dim: int = 256,
        use_transformer: bool = True,
    ):
        """
        Args:
            num_atom_types: Number of atom types
            max_u: Maximum radial degree
            max_v: Maximum angular degree
            num_average_neigh: Average number of neighbors
            max_chebyshev_degree: Maximum degree of Chebyshev polynomial
            r_cut: Cutoff radius
            radial_mlp_hidden_layers: Hidden layers for radial MLP
            mix_atom_feats_radial_channel: Whether to mix input atom features
            max_atom_feats_rank: Maximum rank of input atom features
            max_out_rank: Maximum rank of output features
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            dropout: Dropout rate
            ffn_dim: Feed-forward network dimension in transformer
            use_transformer: Whether to use transformer in this layer
        """
        super().__init__()

        self.num_atom_types = num_atom_types
        self.max_u = max_u
        self.max_v = max_v
        self.num_average_neigh = num_average_neigh
        self.max_chebyshev_degree = max_chebyshev_degree
        self.r_cut = r_cut
        self.radial_mlp_hidden_layers = radial_mlp_hidden_layers
        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer

        self.mix_atom_feats_radial_channel = mix_atom_feats_radial_channel
        self.max_atom_feats_rank = (
            max_v if max_atom_feats_rank is None else max_atom_feats_rank
        )
        self.max_out_rank = max_v if max_out_rank is None else max_out_rank

        # Linear mixing for input atom features
        self.mlp_mix_atom_feats = nn.ModuleDict(
            {
                str(rank): LinearMap(max_u + 1, max_u + 1)
                for rank in range(self.max_atom_feats_rank + 1)
            }
        )

        # CAMP components
        self.atom_moment = AtomicMoment(
            max_u=max_u,
            max_v1=self.max_atom_feats_rank,
            max_v2=max_v,
            num_atom_types=num_atom_types,
            num_average_neigh=num_average_neigh,
            max_chebyshev_degree=max_chebyshev_degree,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            r_cut=r_cut,
        )

        self.hyper_moment = HyperMoment(max_u, max_v, self.max_out_rank)

        # Transformer component for global interactions
        self.use_transformer = use_transformer
        
        # Project scalar features to transformer hidden dimension
        self.scalar_to_hidden = nn.Linear(max_u + 1, hidden_dim)
            
        # Transformer layer for global message passing
        self.transformer = TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                ffn_dim=ffn_dim
            )
            
        # Project back from transformer dimension to scalar features
        self.hidden_to_scalar = nn.Linear(hidden_dim, max_u + 1)

        # Number of radial degrees
        n_u = max_u + 1

        # Params for mixing radial channel of hyper moment
        self.linear_channel_hyper = nn.ModuleDict(
            {str(rank): LinearMap(n_u, n_u) for rank in range(self.max_out_rank + 1)}
        )

        # Params for mixing radial channel of input atom feats
        if mix_atom_feats_radial_channel:
            self.linear_channel_feats = nn.ModuleDict(
                {
                    str(rank): LinearMap(n_u, n_u)
                    for rank in range(self.max_out_rank + 1)
                }
            )
        else:
            self.linear_channel_feats = nn.ModuleDict({})

    def forward(
        self,
        edge_vector: Tensor,
        edge_idx: Tensor,
        atom_type: Tensor,
        atom_feats_in: dict[int, Tensor],
    ) -> dict[int, Tensor]:
        """
        Forward pass of the CAMP-Transformer layer.
        
        Args:
            edge_vector: Edge vectors between atoms
            edge_idx: Edge indices [2, num_edges]
            atom_type: Atom types [num_atoms]
            atom_feats_in: Input atomic features {rank: tensor}
            
        Returns:
            Updated atomic features {rank: tensor}
        """
        # Mix atom features across radial channels
        atom_feats: dict[int, Tensor] = {}
        for v, f in atom_feats_in.items():
            fn: JITInterface = self.mlp_mix_atom_feats[str(v)]
            atom_feats[v] = fn.forward(f)

        # Apply atomic moment operation
        am = self.atom_moment(edge_vector, edge_idx, atom_type, atom_feats)
        
        # Apply hyper moment operation
        hm = self.hyper_moment(am)  # {v: (n_u, n_atoms, 3, 3, ...)}}

        # Mix radial channel of hyper moment
        for rank, m in hm.items():
            fn: JITInterface = self.linear_channel_hyper[str(rank)]
            hm[rank] = fn.forward(m)

        # Apply transformer to scalar features (rank 0) for global interactions
        if self.use_transformer and 0 in hm:
            # Extract scalar features
            scalar_feats = hm[0]  # [n_u, n_atoms]
            
            # Transpose to [n_atoms, n_u]
            scalar_feats = scalar_feats.transpose(0, 1)
            
            # Project to transformer hidden dimension
            hidden = self.scalar_to_hidden(scalar_feats)  # [n_atoms, hidden_dim]
            
            # Apply transformer for global attention
            hidden = self.transformer(hidden)  # [n_atoms, hidden_dim]
            
            # Project back to original feature space
            scalar_feats = self.hidden_to_scalar(hidden)  # [n_atoms, n_u]
            
            # Transpose back to [n_u, n_atoms]
            hm[0] = scalar_feats.transpose(0, 1)

        out = hm

        # Mix radial channel of input atom feats and add to the output
        if self.mix_atom_feats_radial_channel:
            max_rank = min(self.max_atom_feats_rank, self.max_out_rank)
            for rank in range(max_rank + 1):
                fn: JITInterface = self.linear_channel_feats[str(rank)]
                out[rank] = out[rank] + fn.forward(atom_feats[rank])

        return out