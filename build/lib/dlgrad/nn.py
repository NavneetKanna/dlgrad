from tensor import Tensor
import numpy as np

class MLP:
    """
    MLP class is used for initializing the weights and to perfom the 
    weighted sum.

    Args:
    ----
    1) input_data : size of input sample
    2) output_data : number of neurons in the next layer

    Example:
    -------
    Considering the MNIST, the class would be called as 
    fc1 = MLP(28*28, no_of_neurons_in_first_hidden_layer)
    fc2 = MLP(no_of_neurons_in_first_hidden_layer, 10)

    Attributes:
    ----------
    1) weight : A matrix of dimension (output_data, input_data)

    Example:
    -------
    fc1.weight.shape = (no_of_neurons_in_first_hidden_layer, 28*28)
    

    Complete example:
    ----------------
    no_of_neurons_in_first_hidden_layer = 64
    batch_size = 32

    x_train.shape = (32, 784)
    x_train = [x1 x2 x3 . . . x784
               x1 x2 x3 . . . x784
               .
               . ]
    
    fc1 = MLP(784, 64)

    w1.T = (784, 64)
    w1.T = [w1 . . . .
            w2 . . . .
            w3 . . . .
            . 
            .
            w784 . . .] 

    x = fc1(x_train)

    x.shape = (32, 784) @ (784, 64) = (32, 64)
    x = [(x1*w1 + x2*w2 + .... + x784*w784) (output of 2nd neuron) (output of 3rd neuron) ..... (output of the 64th neuron)
         .                                   .                      .                            .         
         .
         .
         ]
    """
    def __init__(self, input_data: int, output_data: int) -> None:
        self.input_data = input_data
        self.output_data = output_data
        # W = [OUTPUT x INPUT]
        self.weight: Tensor = Tensor.uniform((self.output_data, self.input_data))

    # assert shape 
    def __call__(self, data: Tensor) -> Tensor:
        if isinstance(data, Tensor):
            return Tensor.matmul(data, self.weight.T())

    def __repr__(self) -> str:
        return f"MLP layer (input_data={self.input_data}, output_data={self.output_data})"    
