from dlgrad.loss import crossentropy
from dlgrad.tensor import Tensor
from dlgrad.graph import display_graph

lo_val = 0

def train(model, x_train, y_train, flag, lr=1e-3):
    global lo_val

    x = model.forward(x_train, flag)

    loss = crossentropy(x, y_train, flag)
    if lo_val == 0:
        display_graph()
    lo_val += 1

    loss.backward()

    # print("updating parametrs")
    # Update parameters 
    for parameters in Tensor.get_parameters():
        # print("----")
        # print(parameters.tensor.shape)
        # print(parameters.grad.shape)
        parameters.tensor = parameters.tensor - lr*parameters.grad
        # print(parameters.tensor.shape)
        # print(parameters.grad.shape)

    # Calculate accuracy and loss 
    if lo_val == 1874:
     print(f"loss {loss.tensor.mean()}")
     lo_val = 0
