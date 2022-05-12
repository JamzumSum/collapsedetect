"""Detect model collapse for PyTorch module"""

import torch
from torch.jit._trace import trace as ts_trace

T = torch.Tensor


def _max_energy(tensors: T) -> T:
    m = tensors.mean(dim=0)  # [D]
    e: T = (tensors - m).norm(dim=-1, p=2)  # type: ignore
    return e.max()


_max_energy_t = ts_trace(_max_energy, torch.rand((2, 3), dtype=torch.float32))


def is_collapse(*tensors: T, eps: float = 1e-5) -> bool:
    """Detect if the given tensors are collapsed. Requires more than one 1D tensor,
    or 2D tensors with the first dimension greater than 1.

    :param tensors: tensors to be detected. Can be either 1D or 2D. All tensors are unsquashed to 2D (if needed) and are concatenated to a tensor of [N, D], which N > 1.
    :param eps: The result max energy will be compared with this value.
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
    return float(_max_energy_t(tsrs)) < eps
