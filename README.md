# CollapseDetect

_Modle collapse_ is an abnormal and unexpected result in deep learning. This is commonly encountered in metric learning, self-supervised learning, etc.

This package visualize.

[简体中文](README.zh-cn.md)

---

## Install

```sh
pip install collapsedetect git+https://github.com/JamzumSum/collapsedetect.git
```

For `collapsedetect.detector.CollapseVisualizer`, you need to have `tensorboard` installed:

```sh
pip install collapsedetect[tb] git+https://github.com/JamzumSum/collapsedetect.git
```

or you may install `tensorboard` manually :D

## Usage

### core function

```py
>>> from collapsedetect import is_collapse
>>> vectors = normal_network() # (N, D), which N is batchsize and D is vector length
>>> is_collapse(vectors)
False
>>> is_collapse(torch.ones((2, 3), dtype=torch.float32))    # collapsed input
True
>>> is_collapse(torch.randn((3,)), torch.randn((2, 3)), torch.randn((3, 3)))    # multiple input will be concatenated together
False
```

### detector

#### simple usage

```py
det = CollapseDetector()
for epoch in range(15):
    vec = collapse_generator()
    det.append(vec)

assert det.detect() == True
```

#### hooks

```py
det = CollapseDetector()
det.hook(collapse_generator)    # inspect the output of a module
for epoch in range(15):
    collapse_generator()
assert det.detect() == True
det.rm_hook()
```

For more complex usage, see docstrings in `collapsedetect.detector.CollapseDetector.hook`.

## License

- MIT
