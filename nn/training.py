from dlgrad.loss import crossentropy
from dlgrad.afu import softmax
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Optional
from dlgrad.tensor import Tensor
def train(model, train_loader, x_train: np.ndarray, y_train: np.ndarray, BS: int, optimizer, metrics = False) -> Optional[int]:
    acc = None
    loss = None

    steps = train_loader.num_train_steps(BS)
    steps  = 100 

    for _ in (pbar := trange(steps)):
        x_batch_train, y_batch_train = train_loader.get_batch_data(x_train, y_train, BS)
     
        x = model.forward(x_batch_train)
        loss = crossentropy(x, y_batch_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = accuracy_score(y_batch_train.tensor.T, np.argmax(softmax(x), axis=1))
        pbar.set_postfix_str(f"loss {loss.tensor} accuracy {acc*100}%")

    if metrics: return acc, loss.tensor 
        
def plot_metrics(acc_graph, loss_graph):
    fig, ax = plt.subplots()
    ax.plot(loss_graph, label='Loss')
    ax.plot(acc_graph, label='Accuracy')
    ax.legend()
    plt.close(fig)
    fig.savefig("notes/Metrics.png")

def test(model, x_test, y_test):
    x = model.forward(x_test)
    loss = crossentropy(x, y_test)
    print()
    print(f"Test loss : {loss.tensor}")
    acc = accuracy_score(y_test.tensor.T, np.argmax(softmax(x), axis=1))
    print(f"Test accuracy : {acc*100}%")
