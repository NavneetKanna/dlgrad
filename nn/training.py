from dlgrad.loss import crossentropy
from dlgrad.tensor import Tensor
from dlgrad.graph import display_graph
from dlgrad.afu import softmax
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import trange

ff = True
loss_graph = []
acc_graph = []
pred = 0
acc = 0
loss = 0


# TODO: Should we delete all the objects after every loop ?
def train(model, train_loader, x_train: np.ndarray, y_train: np.ndarray, steps: int, BS: int, lr=1e-3):
    global ff, acc_graph, loss_graph, pred, glo_acc, glo_loss

    # for _ in (pbar := tqdm(range(steps))):
    for _ in (pbar := trange(steps)):
    # for i in range(steps):
        x_batch_train, y_batch_train = train_loader.get_batch_data(x_train, y_train, BS)

        
        x = model.forward(x_batch_train)
        pred = x
    
        loss = crossentropy(x, y_batch_train)
        glo_loss = loss
        
        flag = False
    
        if ff: 
            display_graph()
            ff = False

        Tensor.zero_grad()

        loss.backward()

        for parameters in Tensor.get_parameters():
            parameters.tensor = parameters.tensor - (lr*parameters.grad)

        Tensor.save_for_backward.clear()

        glo_acc = accuracy_score(y_batch_train.tensor.T, np.argmax(softmax(pred), axis=1))
        pbar.set_postfix_str(f"loss {glo_loss.tensor} accuracy {glo_acc*100}%")

    acc_graph.append(glo_acc)
    loss_graph.append(glo_loss.tensor)
        
    # print("pred")
    # print(np.argmax(softmax(pred), axis=1))


def draw_cg():
    pass



def plot_metrics():
    global loss_graph, acc_graph

    fig, ax = plt.subplots()
    ax.plot(loss_graph, label='Loss')
    ax.plot(acc_graph, label='Accuracy')
    ax.legend()
    plt.close(fig)
    fig.savefig("Metrics.png")

def test(model, x_test, y_test):
    x = model.forward(x_test)
    loss = crossentropy(x, y_test)
    print()
    print(f"Test loss : {loss.tensor}")
    acc = accuracy_score(y_test.tensor.T, np.argmax(softmax(x), axis=1))
    print(f"Test accuracy : {acc*100}%")
