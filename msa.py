import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional
from torch.nn import LayerNorm
from functools import partial
from typing import Dict, Callable, List, Tuple, Sequence, Union
from functools import partialmethod

def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))

class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            init: str = "default",
            init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
            precision=None
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        # with torch.no_grad():
        #     if init_fn is not None:
        #         init_fn(self.weight, self.bias)
        #     else:
        #         if init == "default":
        #             lecun_normal_init_(self.weight)
        #         elif init == "relu":
        #             he_normal_init_(self.weight)
        #         elif init == "glorot":
        #             glorot_uniform_init_(self.weight)
        #         elif init == "gating":
        #             gating_init_(self.weight)
        #             if bias:
        #                 self.bias.fill_(1.0)
        #         elif init == "normal":
        #             normal_init_(self.weight)
        #         elif init == "final":
        #             final_init_(self.weight)
        #         else:
        #             raise ValueError("Invalid init string.")

        self.precision = precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearNoBias(Linear):
    """
        Convenience class for readability.
    """
    __init__ = partialmethod(Linear.__init__, bias=False)


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
            nn.Sigmoid()
        )

        # Pair
        self.proj_pair_bias = nn.Sequential(
            LayerNorm(c_z),
            LinearNoBias(c_z, no_heads, init="normal")
        )

        # Output projection
        self.output_proj = LinearNoBias(no_heads * c_hidden, c_msa, init='final')

        self.softmax = nn.Softmax(dim=-2)

    def forward(
            self,
            m: Tensor,
            z: Tensor,
            msa_mask: Optional[Tensor] = None,
            z_mask: Optional[Tensor] = None
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
        new_v_shape = (v.shape[:-4] + (n_seq, n_res, n_res, self.no_heads, self.c_hidden))
        v = v.unsqueeze(-4).expand(new_v_shape)  # (*, seq, res, res, heads, c_hidden)

        # Weighted average with gating
        weights = self.softmax(b)
        weights = weights.unsqueeze(-4).unsqueeze(-1)  # (*, 1, res, res, heads, 1)
        o = g * torch.sum(v * weights, dim=-3)  # (*, seq, res, heads, c_hidden)

        # Output projection
        output = self.output_proj(flatten_final_dims(o, 2))  # (*, seq, res, c_hidden * heads)
        return output
    

if __name__ == "__main__":
    N_seq = 3
    N_res = 384
    B = 1
    C_m = 64
    C_z = 128
    
    m = torch.randn((B, N_seq, N_res, C_m))
    z = torch.randn((B, N_res, N_res, C_z))
    
    msa = MSAPairWeightedAveraging(c_msa=64,
            c_z=128,
            c_hidden=32,
            no_heads=8)
    o = msa(m, z)
    print(o)
