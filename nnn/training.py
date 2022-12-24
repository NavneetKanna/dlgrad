from dlgrad.loss import crossentropy
from dlgrad.tensor import Tensor

def train(model, x_train, y_train, lr=1e-3):
    x = model.forward(x_train)
    loss = crossentropy(x, y_train)
    loss.backward()
    for parameters in Tensor.get_parameters():
        parameters.tensor = parameters.tensor - lr*parameters.grad
