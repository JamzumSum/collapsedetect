import torch
from pytest import fixture
from torch.nn import Module


class CollapseModule(Module):
    def forward(self):
        return torch.tensor([1, 1, 1], dtype=torch.float32)


class NormalModule(Module):
    def forward(self):
        return torch.randn((3,), dtype=torch.float32)


@fixture
def collapse_generator():
    return CollapseModule()


@fixture
def normal_generator():
    return NormalModule()
