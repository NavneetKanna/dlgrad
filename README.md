# dlgrad

dlgrad (*D*eep *L*earning auto*grad*): A Lightweight Autograd Engine for Deep Learning

Inspired by Andrej Karpathy's micrograd and George Hotz's tinygrad, dlgrad is my personal exploration into building an Autograd engine from scratch. Its lightweight in design and has PyTorch like API.

## Features

- **CPU and GPU Support**: The library currently supports CPU backend and GPU support is coming in future.

## Internals

You can read my [blog](https://navneetkanna.github.io/blog/2024/02/22/dlgrad-Behind-the-scenes.html) to learn more about how dlgrad operates.

## Things I'm Working On
- [ ] Big change coming to dlgrad

##  History

<details>

<summary> A brief timeline of dlgrad </summary>

- I started this project in 2022 with the intention of learning the fundamentals of deep learning. The initial version worked perfectly fine but was just a numpy wrapper.
- In early 2024, I revisted the project and realised that I didnt learn/do much since most of the heavy lifting was done by numpy and this bothered me.
- Hence, I began to rewrite dlgrad, well, in a stupid way. 
- Since, I didnt want to rely on numpy at all, I needed some way in creating the tensors. My genius idea was, let me write C code in python, compile them as a shared file and load them into python. 

</details>