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
import os

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
    C_hidden, N_head, inf,
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
    offs_j = tl.arange(0, RES_LEN_POW2)
    offs_c = tl.arange(0, C_LEN_POW2)
    
    # Load in b weight i:i+BLOCK_SIZE_ROW and compute softmax
    b_offs = (z_off * RES_LEN * RES_LEN * N_head) + \
            (offs_i[:, None] * RES_LEN * N_head) + \
            (offs_j[None, :] * N_head) + \
            (h_off)
    
    ij_mask = ((offs_i < RES_LEN)[:, None]) & ((offs_j < RES_LEN)[None, :])
    
    b = tl.load(b_ij_ptr + b_offs, ij_mask, -inf)
    
    # Compute softmax of row (assuming all loaded in, should be cheap since w is small)
    # log2_e = 1.44269504
    # exp2 = lambda x : tl.exp2(log2_e * x)
    
    row_minus_max = b - tl.max(b, axis=1, keep_dims=True)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1, keep_dims=True)
    w = numerator / denominator
    
    # Compute output
    for s in range(0, SEQ_LEN, BLOCK_SIZE_SEQ):
        # Offsets for {s,i} indices
        
        for ch in range(0, C_hidden, 1):
            offs_s = s + tl.arange(0, BLOCK_SIZE_SEQ)
            si_off = (z_off * SEQ_LEN * RES_LEN * N_head * C_hidden) + \
                     (offs_s[None, :] * RES_LEN * N_head * C_hidden) + \
                     (offs_i[:, None] * N_head * C_hidden) + \
                     (h_off * C_hidden) + \
                     (ch)
            si_mask = ((offs_s < SEQ_LEN)[None, :]) & ((offs_i < RES_LEN)[:, None])

            # Load in g_{s,i} transposed
            g = tl.load(g_si_ptr + si_off, si_mask, 0)
            g = tl.sigmoid(g)

            sj_off = (z_off * SEQ_LEN * RES_LEN * N_head * C_hidden) + \
                     (offs_s[None, :] * RES_LEN * N_head * C_hidden) + \
                     (offs_j[:, None] * N_head * C_hidden) + \
                     (h_off * C_hidden) + \
                     (ch)
            sj_mask = ((offs_s < SEQ_LEN)[None, :]) & ((offs_j < RES_LEN)[:, None])

            # Load in v_{s,j} transposed
            v = tl.load(v_si_ptr + sj_off, sj_mask, 0)
            
            # W has shape I x J
            vw = tl.zeros_like(g)
            
            vw = tl.dot(w, v, vw) # (I x J) x (J x S) = I x S

            # Element-wise product of output
            out = g * vw
            tl.store(output_ptr + si_off, out, si_mask)


class _MSAWeightedAveragingFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, b, g, inf):
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

        grid = (
            n_batches,
            no_heads,
            triton.cdiv(n_res, BLOCK_SIZE_ROW),
        )

        MSAFwdFused[grid](
            v, b, g, out, 
            C_hidden, no_heads, inf,
            c_hidden_pow2, n_res_pow2, 
            n_seq, n_res,
            BLOCK_SIZE_ROW,
            BLOCK_SIZE_SEQ,
        )
        
        return out

    @staticmethod
    def backward(ctx, do):
        pass

MSAWeightedAveragingFused = _MSAWeightedAveragingFused.apply
