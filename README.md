<div align="center">

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NavneetKanna/dlgrad)

</div>

# dlgrad

dlgrad (*D*eep *L*earning auto*grad*) is a small but highly capable reverse-mode autodiff engine with a PyTorch-like API.

It lives in the sweet spot between micrograd and tinygrad — simple enough to learn from, powerful enough to run real models.

Built from scratch to learn how deep learning and ML frameworks work.

## Installation

```bash
python3 -m pip install git+https://github.com/NavneetKanna/dlgrad
```

## Documentation

Documentation can be found in this [folder](docs/README.md).

## Examples

The mnist MLP and GAN examples can be found in the [examples folder](examples/).

The mnist example gets to around 95% accuracy in ~5 seconds on the CPU.

## Tests

To run the tests (pytest is required)

```bash
python3 -m pytest test/
```

To compare the speed of dlgrad with pytorch and tinygrad
```python3
OMP_NUM_THREADS=1 python3 -m pytest -s -q -k test_speed_v_torch
```
