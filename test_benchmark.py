# Author: Alex Zhang

import torch
import pytest
import triton

from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional
from torch.nn import LayerNorm
from functools import partial
from typing import Dict, Callable, List, Tuple, Sequence, Union
from functools import partialmethod
from utils import flatten_final_dims, Linear, LinearNoBias
from msa import MSAPairWeightedAveraging, MSAWeightedAveragingNaive
from msa_kernel import MSAWeightedAveragingFused
    
# TODO: CHANGE BACK TO 2, 8, 32, 64, 128, 128, 32
@pytest.mark.parametrize("B, H, N_seq, N_res, C_m, C_z, C_hidden", [(2, 8, 32, 64, 128, 128, 32)])
def test_correctness(B, H, N_seq, N_res, C_m, C_z, C_hidden, dtype=torch.float32):
    torch.manual_seed(20)
  
    m = torch.randn((B, N_seq, N_res, C_m), dtype=dtype, device="cuda").requires_grad_()
    z = torch.randn((B, N_res, N_res, C_z), dtype=dtype, device="cuda").requires_grad_()
    
    msa = MSAPairWeightedAveraging(
            c_msa=C_m,
            c_z=C_z,
            c_hidden=C_hidden,
            no_heads=H
          ).to("cuda")
    
    ref_out = msa(m, z, use_triton_kernel=False)
    tri_out = msa(m, z, use_triton_kernel=True)
    
    # Compare FWD pass
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    rtol = 0.0

    
    ### TESTING GRADIENTS
    v = torch.randn((B, N_seq, N_res, H, C_hidden), dtype=dtype, device="cuda").requires_grad_()
    b = torch.randn((B, N_res, N_res, H), dtype=dtype, device="cuda").requires_grad_()
    g = torch.randn((B, N_seq, N_res, H, C_hidden), dtype=dtype, device="cuda").requires_grad_()
    ref_msa = MSAWeightedAveragingNaive(H, C_hidden)
    ref_out_kernel = ref_msa(v, b, g, N_seq, N_res)
    tri_out_kernel = MSAWeightedAveragingFused(v, b, g)
    
    dout = torch.randn_like(ref_out_kernel)
    
    ref_out_kernel.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_db, b.grad = b.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    
    
    tri_out_kernel.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_db, b.grad = b.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2

    assert torch.allclose(ref_dg, tri_dg, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_db, tri_db, atol=1e-2, rtol=rtol)


TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, N_SEQ, C_HIDDEN, C_m, C_z = 1, 8, 64, 32, 128, 128
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_res"],
            x_vals=[32, 64] + [128 * (k+1) for k in range(50)], # 384, 768, 1536, 3072
            line_arg="provider",
            line_vals=["triton_msa"] + ["baseline"],
            line_names=["Triton_msa"] + ["Baseline"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="Runtime (ms)",
            plot_name=f"msa-batch={BATCH}-head={N_HEADS}-c={C_HIDDEN}-N_seq={N_SEQ}-C_m={C_m}-C_z={C_z}-{mode}",
            args={
                "N_HEADS": N_HEADS,
                "BATCH": BATCH,
                "N_seq": N_SEQ,
                "C_HIDDEN": C_HIDDEN,
                "C_m": C_m,
                "C_z": C_z,
                "mode": mode,
            },
            ))


@triton.testing.perf_report(configs)
def bench_msa(BATCH, N_res, N_HEADS, N_seq, C_HIDDEN, C_m, C_z, mode, provider, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    dtype = torch.float32
    
    m = torch.randn((BATCH, N_seq, N_res, C_m), dtype=dtype, device="cuda").requires_grad_()
    z = torch.randn((BATCH, N_res, N_res, C_z), dtype=dtype, device="cuda").requires_grad_()
    
    msa = MSAPairWeightedAveraging(
            c_msa=C_m,
            c_z=C_z,
            c_hidden=C_HIDDEN,
            no_heads=N_HEADS
          ).to(device)
    
    try: 
        if "triton_msa" in provider:
            fn = lambda: msa.forward(m, z, use_triton_kernel=True)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)

        elif "baseline" in provider:
            fn = lambda: msa.forward(m, z, use_triton_kernel=False)
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    # CUDA OOM
    except: 
        ms = None

    return ms


if __name__ == "__main__":
    bench_msa.run(save_path="data/", print_data=True)
