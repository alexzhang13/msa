import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch.nn import LayerNorm
from functools import partial
from typing import Dict, Callable, List, Tuple, Sequence, Union
from functools import partialmethod
import math

import triton
import triton.language as tl

from utils import flatten_final_dims, Linear, LinearNoBias

def nearest_pow2(n: int):
    power = math.ceil(math.log2(n))
    next_power_of_two = 2 ** power
    return next_power_of_two

@triton.jit
def MSAFwdFused(
    v_si_ptr, b_ij_ptr, g_si_ptr, output_ptr,
    C_hidden, N_head,
    C_LEN_POW2: tl.constexpr,
    RES_LEN_POW2: tl.constexpr,
    SEQ_LEN: tl.constexpr, RES_LEN: tl.constexpr, 
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
):
    # Compute the program ID and starting index
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)
    
    z_off = pid_z.to(tl.int64)
    h_off = pid_h.to(tl.int64)
    i_off = pid_i.to(tl.int64) * BLOCK_SIZE_ROW
    offs_i = i_off + tl.arange(0, BLOCK_SIZE_ROW)
    offs_c = tl.arange(0, C_LEN_POW2)
    
    # Load in b weight i:i+BLOCK_SIZE_ROW and compute softmax
    offs_j = tl.arange(0, RES_LEN_POW2)
    b_off = (offs_i[:, None] * RES_LEN * N_head) + \
            (offs_j[None, :] * N_head) + \
            (h_off)
    ij_mask = ((offs_i < RES_LEN)[:, None]) & ((offs_j < RES_LEN)[None, :])
    
    b = tl.load(b_ij_ptr + b_off, ij_mask, 0)
    
    # Compute softmax of row (assuming all loaded in)
    row_minus_max = b - tl.max(b, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    w = numerator / denominator
    w = tl.broadcast_to(tl.trans(w, 1, 0).expand_dims(0), BLOCK_SIZE_SEQ, RES_LEN, BLOCK_SIZE_ROW)
    
    # Compute output
    for s in range(0, SEQ_LEN, BLOCK_SIZE_SEQ):
        # Offsets for {s,i} indices
        offs_s = s + tl.arange(0, BLOCK_SIZE_SEQ)
        si_off = (z_off * SEQ_LEN * RES_LEN * N_head * C_hidden) + \
                 (offs_s[:, None, None] * RES_LEN * N_head * C_hidden) + \
                 (offs_i[None, :, None] * N_head * C_hidden) + \
                 (h_off * C_hidden) + \
                 (offs_c[None, None, :])
        si_mask = ((offs_s < SEQ_LEN)[:, None, None]) & ((offs_i < RES_LEN)[None, :, None]) & ((offs_c < C_hidden)[None, None, :])
        
        # Load in g_{s,i}
        g = tl.load(g_si_ptr + si_off, si_mask, 0)
        g = tl.sigmoid(g)
        
        sj_off = (z_off * N_head * SEQ_LEN * RES_LEN * C_hidden) + \
                    (offs_s[:, None, None] * RES_LEN * N_head * C_hidden) + \
                    (offs_j[None, :, None] * N_head * C_hidden) + \
                    (h_off * C_hidden) + \
                    (offs_c[None, None, :])
        sj_mask = ((offs_s < SEQ_LEN)[:, None, None]) & ((offs_j < RES_LEN)[None, :, None]) & ((offs_c < C_hidden)[None, None, :])
        
        # Load in v_{i,j}
        v = tl.load(v_si_ptr + sj_off, sj_mask, 0)
        
        v = tl.trans(v, 0, 2, 1)
        vw = tl.trans(tl.dot(v, w), 0, 2, 1)

        # Element-wise product of output
        out = vw # g * vw
        tl.store(output_ptr + si_off, out, si_mask)


class _MSAWeightedAveragingFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, b, g):
        """
        Fuse the softmax and linear combination step of MSA.
        """
        n_batches, n_seq, n_res, no_heads, C_hidden = v.shape
        
        # allocate output
        out = torch.empty((n_batches, n_seq, n_res, no_heads * C_hidden), device=g.device, dtype=g.dtype)
        
        BLOCK_SIZE_ROW = 16
        BLOCK_SIZE_SEQ = 16
        n_res_pow2 = nearest_pow2(n_res)
        c_hidden_pow2 = nearest_pow2(C_hidden)

        # output = torch.empty((N, S, C), device=m_si.device, dtype=m_si.dtype)

        grid = (
            n_batches,
            no_heads,
            triton.cdiv(n_res, BLOCK_SIZE_ROW),
        )

        MSAFwdFused[grid](
            v, b, g, out,
            C_hidden, no_heads,
            c_hidden_pow2,
            n_res_pow2, n_seq, n_res,
            BLOCK_SIZE_ROW,
            BLOCK_SIZE_SEQ,
        )
        
        return out

    @staticmethod
    def backward(ctx, do):
        pass

MSAWeightedAveragingFused = _MSAWeightedAveragingFused.apply
    
class MSAPairWeightedAveragingFused(nn.Module):
    def __init__(
        self,
        c_msa: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        inf: float = 1e8
    ):
        super(MSAPairWeightedAveragingFused, self).__init__()
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

    def forward(
        self,
        m: Tensor,
        z: Tensor,
        msa_mask: Optional[Tensor] = None,
        z_mask: Optional[Tensor] = None,
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

        print('g shape', g.shape)
        print('g', g)
        
        del m_ln

        # Masking and shape wrangling
        if z_mask is not None:
            z_mask = z_mask.unsqueeze(-1)  # (*, N_res, N_res, 1)
            z_mask = self.inf * (z_mask - 1)  # mask before softmax
            b = b + z_mask

        if msa_mask is not None:
            v = v * msa_mask.unsqueeze(-1).unsqueeze(-1)
        print('b shape', b.shape)
        
        # Weighted average with gating
        o = MSAWeightedAveragingFused(v, b, g)
        
        print('out', o)

        # Output projection
        output = self.output_proj(o)  # (*, seq, res, c_msa)
        return output
    
if __name__ == "__main__":
    N_seq = 3
    N_res = 128 # 384
    B = 1
    C_m = 64
    C_z = 128
    C_hidden = 32
    no_heads = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    m = torch.randn((B, N_seq, N_res, C_m), device=device)
    z = torch.randn((B, N_res, N_res, C_z), device=device)
    
    msa = MSAPairWeightedAveragingFused(
            c_msa=C_m,
            c_z=C_z,
            c_hidden=C_hidden,
            no_heads=no_heads
          ).to(device)
    
    o = msa(m, z)
    # print('projected', o)
