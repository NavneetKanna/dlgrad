from tqdm import tqdm

from dlgrad import Tensor, nn
from dlgrad.nn.datasets import mnist

BS, in_dim, HS, ncls = 128, 784, 64, 10
EPOCHS = 3
STEPS = int(60000/BS) * EPOCHS

device = "cpu" # or metal

class Model:
    def __init__(self) -> None:
        self.layers = [
            nn.Linear(in_dim, HS, bias=True, device=device),
            Tensor.relu,
            nn.Linear(HS, ncls, bias=True, device=device)
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)

# Load MNIST
x_train_images, x_train_labels, x_test_images, x_test_labels = mnist(device=device)

# Normalize to [0, 1]
x_train_images = x_train_images / 255.0
x_test_images = x_test_images / 255.0

model = Model()

opt = nn.optim.SGD(params=nn.utils.get_parameters(model), lr=1e-3) # or Adam

for i in nn.utils.get_parameters(model):
    print(i)

for epoch in range(EPOCHS):
    s = 0
    h = BS

    for i in tqdm(range(STEPS)):
        opt.zero_grad()

        output = model(x=x_train_images[s:h])
        loss = output.cross_entropy_loss(target=x_train_labels[s:h])

        loss.backward()
        opt.step()
        exit()

        # print(epoch, i, s, h)
        s += BS
        h += BS
        if s >= 60000 or h >= 60000:
            s = 0
            h = BS

    # Evaluate
    out = model(x=x_test_images)
    y_pred = out.argmax(dim=1)
    correct = (y_pred==x_test_labels).sum()
    total = float(x_test_labels.shape[0])
    test_acc = ((correct/total) * 100.0)
    print("test acc")
    test_acc.show()
