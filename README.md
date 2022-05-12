# CollapseDetect

_Model collapse_ means model outputs collapse to the same pattern (or maybe just the same output for every input). This is encountered in metric learning, self-supervised learning, etc.

This package include a snippet for detecting output collapse. It also provides a shortcut to visualize output vectors in tensorboard.

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

See our [docs][docs].

## License

- MIT

[docs]: https://jamzumsum.github.io/collapsedetect/
