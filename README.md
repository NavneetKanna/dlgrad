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
- In early 2024, I revisted the project and realised that I didnt learn or do much since most of the heavy lifting was done by numpy and this bothered me.
- Hence, I began to rewrite dlgrad, well, in a stupid way. 
- Since, I didnt want to rely on numpy at all, I needed some way in creating the tensors. My genius idea was, let me write C code in python, compile them as a shared file (using subprocess) and load them into python. Suprisingly it worked. 
- However, it was really difficult to manage creating tensors in C and using them in python. Since the tensors were C arrays, I had to ensure to clean them up, indexing, etc. And I spent around 8 months doing this. Yea 8 months !!!.
- Things were only getting complicated as I sarted to add new ops, losses, etc, and at this point I became frustated at myself, saddend by the fact that I am not able to do this.
- Then I was looking at [llm.c](https://github.com/karpathy/llm.c), and I wondered, why am I complicating things, why am I using python, why cant I just use C. Afterall I am not building a framework to compete with the existing ones, rather the goal was to understand the fundamentals of deep learning, which can be done in C also, as demonstrated by llm.c

</details>