"""Detect model collapse for PyTorch module"""

import torch
from torch.jit._trace import trace as ts_trace

T = torch.Tensor


def _energy(tensors: T) -> T:
    m = tensors.mean(dim=0)  # [D]
    return (tensors - m).norm(dim=-1, p=2)  # type: ignore


_energy_t = ts_trace(_energy, torch.rand((2, 3), dtype=torch.float32))


def is_collapse(*tensors: T, eps: float = 1e-5, tolerance: float = 0.1) -> bool:
    r"""Detect if the given tensors are collapsed. Requires more than one 1D tensor,
    or 2D tensors with the first dimension greater than 1.

    :math:`is\_collapse(x_{ij}, \varepsilon, \tau) = {\underset {0 \leq i < B} {countif}}(||x_i - \bar x||_2 > \varepsilon) \geq 1 - \tau`

    :param tensors: tensors to be detected. Can be either 1D or 2D. All tensors are unsquashed to 2D (if needed) and are concatenated to a tensor of [N, D], which N > 1.
    :param eps: :math:`\varepsilon` in the defination above. The norm results will be compared with this value, defaults to `1e-5`.
    :param tolerance: :math:`\tau` in the defination above. Tolerance rate, default as 0.1
    :return: whether input tensors are collapsed.

    >>> is_collapse(torch.ones((2, 3), dtype=torch.float32))
    True
    >>> is_collapse(torch.randn((3,)), torch.randn((3,)))
    False
    """
    tsrs = []
    for i in tensors:
        if i.ndim == 1:
            tsrs.append(i.detach().cpu().unsqueeze(0))
        elif i.ndim == 2:
            tsrs.append(i.detach().cpu())
        else:
            raise ValueError(
                f"Expect input tensors have 1/2 ndim. But got ndim={i.ndim}."
            )

    tsrs = torch.cat(tsrs)
    assert tsrs.size(0) > 1, "requires more than one 1D tensors"
    ba: T = _energy_t(tsrs.float()) < eps
    return float(ba.sum() / ba.size(0)) + tolerance >= 1
