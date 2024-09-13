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
