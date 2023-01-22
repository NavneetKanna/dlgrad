from __future__ import annotations
import numpy as np
from .graph import draw_graph, CG



def _add_matmul_node_once(output, matrix):
    static_var = _add_matmul_node_once.static_var
    if not static_var:
        _add_matmul_node_once.static_var = True

_add_matmul_node_once.static_var = False


# TODO: Arrange ops
class Tensor:
    save_for_backward = []
    weight_parameters = []
    bias_parameters = []
    parameters = []
    cg = CG()

    def __init__(self, data: np.ndarray):
        self.tensor: np.ndarray = data 
        # Check first if it has tensor attribute
        self.shape = self.tensor.shape
        self._backward = lambda: None
        self.grad = np.zeros(self.shape)

    @classmethod
    def uniform(cls, shape: tuple):
        # The new way 
        return cls(np.random.default_rng().uniform(-1, 1, shape))

    @classmethod
    def zeros(cls, shape: tuple):
        return cls(np.zeros(shape))

    # UnaryOps
    # def T(self):
    #     return Tensor(self.tensor.T)

    def max(data, axis):
        return np.max(data.tensor, axis=axis)
    
    # BinaryOps
    def add(self: Tensor, other: Tensor, flag=None):
        output = Tensor(self.tensor + other.tensor)
        Tensor.save_for_backward.extend((self, other))

        if not CG.stop_processing: CG.add_nodes('add', output.tensor, self.tensor, other.tensor)

        # if flag:
        #     draw_graph(
        #         'add',
        #         (output.tensor.shape, 'output'),
        #         (self.tensor.shape, 'input 1'), 
        #         (other.tensor.shape, 'input 2')
        #     )
            
        def backward():
            self.grad =  output.grad
            other.grad =  np.sum(output.grad, axis=0, keepdims=True)
        
        output._backward = backward

        return output

    def sub(self, other):
        return Tensor(self.tensor-other.tensor)

    def matmul(self: Tensor, other: Tensor, flag=None):
        output = Tensor(self.tensor @ other.tensor.T)
        Tensor.save_for_backward.extend((self, other))

        if not CG.stop_processing: CG.add_nodes('matmul', output.tensor, self.tensor, other.tensor)

        # if flag:
        #     draw_graph(
        #         'matmul',
        #         (output.tensor.shape, 'output'),
        #         (self.tensor.shape, 'input 1'),
        #         (other.tensor.shape, 'input 2')
        #     )

        def backward():
            self.grad =  (output.grad @ other.tensor)
            other.grad =  (self.tensor.T @ output.grad).T

        output._backward = backward

        return output 

    def __repr__(self) -> str:
        return f"Tensor({self.tensor})"

    def __setitem__(self, key, val):
        self.tensor[key] = [val for i in self.tensor if i in key]

    def __getitem__(self, val) -> np.ndarray:
        return self.tensor[val]

    def get_weight_parameters():
        return Tensor.weight_parameters 
    
    def get_bias_parameters():
        return Tensor.bias_parameters
    
    def get_parameters():
        return Tensor.parameters
    
    @staticmethod
    def zero_grad():
        for parameters in Tensor.save_for_backward:
            parameters.grad = np.zeros(parameters.shape) 
        
    # Autograd Engine

    # As of now I dont see any need for topological sort as others have used it. 
    # The save_for_backward list contians the nodes already ordered since we build
    # the neural network architecture sequentially.
    def backward(self):
        Tensor.save_for_backward.append(self)
        for node in reversed(Tensor.save_for_backward):
            node._backward() 

