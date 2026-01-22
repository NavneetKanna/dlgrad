<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="assets/dlgrad_logo_dark.svg">
  <img alt="dlgrad logo" src="assets/dlgrad_logo_light.svg" width="50%" height="50%">
</picture>

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NavneetKanna/dlgrad)

</div>

---

# dlgrad

dlgrad (*D*eep *L*earning auto*grad*) is a small but capable reverse-mode autodiff engine with a PyTorch-like API.

It sits in the sweet spot between micrograd and tinygrad — simple enough to study, yet powerful enough to run real models.

Built from scratch to learn how deep learning and ML frameworks work. It is not intended to replace existing libraries, but to learn how they operate and to understand how deep learning models train/learn.

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

[examples folder](examples/) contains the implementations of MLP, GAN and GPT.

A GPT Model implemented from scratch, capable of generating English text. To run

```python3
# Train the model (Automatically downloads dataset)
python3 gpt.py --train

# Run inference (Automatically downloads weights, infinite generation)
python3 gpt.py --infer
```

The final loss reached was around `1.28`. The trained model parameters are 

```
vocab_size = 0
block_size = 128 # Context length
n_layer = 6
n_head = 4
n_embd = 128
dropout = 0.2
learning_rate = 1e-4
max_iters = 10000
batch_size = 16
eval_interval = 500
device = "cpu"
```

The MNIST example gets to around 95% accuracy in ~5 seconds on an M2 CPU.

## Features

- Reverse-mode autodiff with broadcasting support.
- Auto-generated shape-specialized kernels.
- Supports CPU and Metal accelerators.
- Lightweight and fast.
- Trains real models (MNIST ~95% in ~5s on M2 CPU)
- Only three dependencies: cffi, pyobjc, tqdm.

## Tests

To run the tests

```bash
python3 -m pip install -e '.[testing]'
python3 -m pytest test/
```

To compare the speed of dlgrad with pytorch and tinygrad
```python3
python3 -m pytest -s test/speed/test_speed_v_torch_tinygrad.py
```

## Further features to add

- [ ] Add a memory checker. Since arrays are created in C, include checks to ensure that dlgrad is not accidentally reading or writing out of bounds.
- [ ] Optimize kernels.
- [ ] Add support for visualization networks graphs.
- [x]  Begin adding support for transformers and large language models, with the long-term goal of fully implementing LLMs. 
- [ ] Add CUDA support.
- [ ] Add batch normalization, additional activation functions, optimizers.
