import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional
from torch.nn import LayerNorm
from functools import partial
from typing import Dict, Callable, List, Tuple, Sequence, Union
from functools import partialmethod
from utils import flatten_final_dims, Linear, LinearNoBias
from msa_kernel import MSAWeightedAveragingFused

class MSAWeightedAveragingNaive(nn.Module):
    def __init__(self, no_heads: int, c_hidden: int):
        super(MSAWeightedAveragingNaive, self).__init__()
        self.no_heads = no_heads
        self.c_hidden = c_hidden
        self.softmax = nn.Softmax(dim=-2)
    
    def forward(self, v, b, g, n_seq, n_res):
        new_v_shape = (v.shape[:-4] + (n_seq, n_res, n_res, self.no_heads, self.c_hidden))
        v = v.unsqueeze(-4).expand(new_v_shape)  # (*, seq, res, res, heads, c_hidden)

        # Weighted average with gating
        weights = self.softmax(b)
        weights = weights.unsqueeze(-4).unsqueeze(-1)  # (*, 1, res, res, heads, 1)
        o = F.sigmoid(g) * torch.sum(v * weights, dim=-3)  # (*, seq, res, heads, c_hidden)
        o = flatten_final_dims(o, 2)
        
        return o

class MSAPairWeightedAveraging(nn.Module):
    def __init__(
            self,
            c_msa: int,
            c_z: int,
            c_hidden: int,
            no_heads: int,
            inf: float = 1e8
    ):
        super(MSAPairWeightedAveraging, self).__init__()
        self.c_msa = c_msa
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        # MSA
        self.msa_ln = LayerNorm(c_msa)
        split_heads = nn.Unflatten(dim=-1, unflattened_size=(no_heads, c_hidden))
        self.msa_proj = nn.Sequential(
            LinearNoBias(c_msa, c_hidden * no_heads, init='glorot'),
            split_heads  # split the heads
        )
        self.to_gamma = nn.Sequential(
            LinearNoBias(c_msa, c_hidden * no_heads, init='gating'),
            split_heads,  # split the heads
        )

        # Pair
        self.proj_pair_bias = nn.Sequential(
            LayerNorm(c_z),
            LinearNoBias(c_z, no_heads, init="normal")
        )

        # Output projection
        self.output_proj = LinearNoBias(no_heads * c_hidden, c_msa, init='final')
        
        # Naive MSA
        self.msa = MSAWeightedAveragingNaive(no_heads, c_hidden)
        

    def forward(
            self,
            m: Tensor,
            z: Tensor,
            msa_mask: Optional[Tensor] = None,
            z_mask: Optional[Tensor] = None,
            use_triton_kernel: bool = False,
    ) -> Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embeddings
            z:
                [*, N_res, N_res, C_z] pair embeddings
            msa_mask:
                [*, N_seq, N_res] MSA mask
            z_mask:
                [*, N_res, N_res] pair mask
        Returns:
            [*, N_seq, N_res, C_m] updated MSA representation
        """
        *_, n_seq, n_res, _ = m.shape

        # Input projections
        m_ln = self.msa_ln(m)
        v = self.msa_proj(m_ln)  # (*, seq, res, heads, c_hidden)
        b = self.proj_pair_bias(z)  # (*, res, res, no_heads)
        g = self.to_gamma(m_ln)  # (*, seq, res, heads, c_hidden)

        del m_ln

        # Masking and shape wrangling
        if z_mask is not None:
            z_mask = z_mask.unsqueeze(-1)  # (*, N_res, N_res, 1)
            z_mask = self.inf * (z_mask - 1)  # mask before softmax
            b = b + z_mask

        if msa_mask is not None:
            v = v * msa_mask.unsqueeze(-1).unsqueeze(-1)
            
        if use_triton_kernel:
            o = MSAWeightedAveragingFused(v, b, g)
        else:
            o = self.msa(v, b, g, n_seq, n_res)

        # Output projection
        output = self.output_proj(o)
        
        return output
    

if __name__ == "__main__":
    N_seq = 16
    N_res = 16
    B = 1
    C_m = 32
    C_z = 128
    C_hidden = 32
    no_heads = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    m = torch.randn((B, N_seq, N_res, C_m), device=device)
    z = torch.randn((B, N_res, N_res, C_z), device=device)
    
    msa = MSAPairWeightedAveraging(
            c_msa=C_m,
            c_z=C_z,
            c_hidden=C_hidden,
            no_heads=no_heads
          ).to(device)
    
    o_o = msa(m, z, use_triton_kernel=False)
    o = msa(m, z, use_triton_kernel=True)

    print(torch.mean(torch.abs(o_o - o)))
