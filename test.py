from dlgrad import Tensor, nn

a = nn.Linear(2, 3)

inp = Tensor.rand(4, 2)

a(inp)