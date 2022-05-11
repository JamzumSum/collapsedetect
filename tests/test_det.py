from pathlib import Path
from shutil import rmtree

import pytest
from torch.nn import Module

from collapsedetect.detector import CollapseDetector, CollapseVisualizer


def test_detector(collapse_generator: Module, normal_generator: Module):
    det = CollapseDetector()
    det.hook(collapse_generator)
    for epoch in range(3):
        collapse_generator()
    assert det.detect() == True
    det.rm_hook()

    det.hook(normal_generator)
    for epoch in range(3):
        normal_generator()
    assert det.detect() == False
    det.rm_hook()


def test_visualizer(collapse_generator: Module):
    try:
        from torch.utils.tensorboard.writer import SummaryWriter
    except ImportError as e:
        pytest.xfail(str(e))

    logdir = Path("log")
    prev = set(logdir.glob("events.out.tfevents.*"))

    with SummaryWriter(logdir) as f:
        viz = CollapseVisualizer(f, "ice.cream")
        viz.hook(collapse_generator)
        for epoch in range(3):
            collapse_generator()
        assert viz.detect() == True

    assert set(logdir.glob("events.out.tfevents.*")) - prev
    rmtree(logdir)
    