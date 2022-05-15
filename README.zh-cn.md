# CollapseDetect

_Model collapse_ means model outputs collapse to the same pattern (or maybe just the same output for every input). This is encountered in metric learning, self-supervised learning, GAN, etc.

This package includes a snippet for detecting output collapse. It also provides a shortcut to visualize output vectors in tensorboard.

[English](README.md)

---

## 安装

```sh
pip install collapsedetect git+https://github.com/JamzumSum/collapsedetect.git
```

欲使用 `collapsedetect.detector.CollapseVisualizer`, 你需要确保安装 `tensorboard`:

```sh
pip install collapsedetect[tb] git+https://github.com/JamzumSum/collapsedetect.git
```

或者你也可以自行安装 `tensorboard` :D

## 如何使用

请移步[文档][docs].

## 核心函数的数学表达

$$
is\_collapse(x_{ij}, \varepsilon, \tau) = {\underset {0 \leq i < B} {countif}}(||x_i - \bar x||_2 > \varepsilon) \geq 1 - \tau
$$

其中 _B_ 为 _batch size_.

## License

- MIT

[docs]: https://jamzumsum.github.io/collapsedetect/
