<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/dlgrad_logo.png">
</picture>

</div>

---

<div align="center">

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NavneetKanna/dlgrad)

</div>

# dlgrad

dlgrad (*D*eep *L*earning auto*grad*) is a small but highly capable reverse-mode autodiff engine with a PyTorch-like API.

It lives in the sweet spot between micrograd and tinygrad — simple enough to learn from and powerful enough to run real models.

Built from scratch to learn how deep learning and ML frameworks work. Note that this is not meant as a replacement to the existing frameworks, rather to learn how they operate and to understand how deep learning models train/learn.

## Installation

From source 

```bash
git clone https://github.com/NavneetKanna/dlgrad.git
cd dlgrad
python3 -m pip install -e .
```

Direct

```bash
python3 -m pip install git+https://github.com/NavneetKanna/dlgrad
```

## Documentation

Documentation can be found in this [folder](docs/README.md).

## Examples

The mnist MLP and GAN examples can be found in the [examples folder](examples/).

```python3
python3 examples/mnist_mlp.py
python3 examples/mnist_gan.py
```

The mnist example gets to around 95% accuracy in ~5 seconds on the CPU.

## Tests

To run the tests (pytest is required)

```bash
python3 -m pytest test/
```

To compare the speed of dlgrad with pytorch and tinygrad
```python3
python3 -m pytest -s test/speed/test_speed_v_torch_tinygrad.py
```
