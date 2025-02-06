# dlgrad

dlgrad (*D*eep *L*earning auto*grad*): A Lightweight Autograd Engine for Deep Learning

Inspired by Andrej Karpathy's micrograd and George Hotz's tinygrad, dlgrad is my personal exploration into building an Autograd engine from scratch. Its lightweight in design and has PyTorch like API.

## Installation

```bash
python3 -m pip install git+https://github.com/NavneetKanna/dlgrad
```

## Internals

You can read my [blog](https://navneetkanna.github.io/blog/2024/02/22/dlgrad-Behind-the-scenes.html), where I document my journey of building dlgrad, sharing insights and explaining some of the ideas/algorithms.

## Example

```python
from dlgrad.tensor import Tensor

a = Tensor.rand((2, 3), requires_grad=True)
b = Tensor.rand((1, 3), requires_grad=True)

c = a + b

c.sum().backward()

print(a.grad.numpy())
print(b.grad.numpy())

```

## Things I'm Working On
- [x] nn
- [x] Cross-Entropy Loss
- [ ] Indexing
 

## Tests

To run the tests (pytest is required)

```bash
python3 -m pytest test/
```

##  History

<details>

<summary> A brief timeline of dlgrad </summary>

- I started this project in 2022 with the intention of learning the fundamentals of deep learning. The initial version worked perfectly fine but was just a numpy wrapper.
- In early 2024, I revisted the project and realised that I didnt learn or do much since most of the heavy lifting was done by numpy and this bothered me.
- Hence, I began to rewrite dlgrad, well, in a stupid way. 
- Since, I didnt want to rely on numpy at all, I needed some way of creating the tensors. My genius idea was, let me write C code in python, compile them as a shared file (using subprocess) and load them into python. Suprisingly it worked. The rational was, I wanted *dlgrad* to be a simple pip install, and didnt want to deal with compiling C code.
- However, it was becoming really difficult to manage tensors in C and using them in python. Things were only getting complicated as I sarted to add new ops, losses, etc. And I spent around 8 months doing this. Yea 8 months !!!.
-  At this point I became frustated at myself, saddend by the fact that I am not able to do this.
- Then I was looking at [llm.c](https://github.com/karpathy/llm.c), and I wondered, why am I complicating things. All this complexity was arising from the fact that I didnt want to compile C code when installing. But, by doing that, I will drasctically improve performance, increase speed and reduce complexity. 
- I am not worried about the time since, as Andrej Karpathy mentions in the Lex podcast, these are just scar tissues. I have learnt from the mistake and hopefully will not repeat it in the future :). Hence, the lesson learnt here is that,      
    - **Don't complicate things**
    - **Before starting out on a project, layout a plan, figure out how you are going to do things beforehand, so that in the future, after putting so much effort on something, it should not come to a hault, because, you didnt think it through enough**.
</details>