# dlgrad

dlgrad (*D*eep *L*earning auto*grad*): A Lightweight Autograd Engine for Deep Learning

Inspired by Andrej Karpathy's micrograd and George Hotz's tinygrad, dlgrad is my personal exploration into building an Autograd engine from scratch. Its lightweight in design and has PyTorch like API.

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
- Since, I didnt want to rely on numpy at all, I needed some way of creating the tensors. My genius idea was, let me write C code in python, compile them as a shared file (using subprocess) and load them into python. Suprisingly it worked. The rational was, I wanted *dlgrad* to be a simple pip install, and didnt want to deal with compiling C code.
- However, it was becoming really difficult to manage tensors in C and using them in python. Things were only getting complicated as I sarted to add new ops, losses, etc. And I spent around 8 months doing this. Yea 8 months !!!.
-  At this point I became frustated at myself, saddend by the fact that I am not able to do this.
- Then I was looking at [llm.c](https://github.com/karpathy/llm.c), and I wondered, why am I complicating things. All this complexity was arising from the fact that I didnt want to compile C code when installing. But, by doing that I will drasctically improve performance, increase speed and reduce complexity. 
</details>