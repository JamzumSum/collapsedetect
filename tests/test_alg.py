import torch
from torch.nn import Module

from collapsedetect import is_collapse


def test_is_collapse(collapse_generator: Module, normal_generator: Module):
    assert True == is_collapse(collapse_generator(), collapse_generator())
    assert False == is_collapse(normal_generator(), normal_generator())
