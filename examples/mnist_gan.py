# ruff: noqa
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dlgrad import Tensor, nn
from dlgrad.nn.datasets import mnist

# parameters
EPOCHS = 300
BS = 100
K = 1
STEPS = 70000 // BS
SAMPLE_INTERVAL = EPOCHS // 10
device = "cpu"

class LinearGen:
    def __init__(self) -> None:
        self.net = [
            nn.Linear(128, 256),
            lambda t: t.leaky_relu(0.2),
            nn.Linear(256, 512),
            lambda t: t.leaky_relu(0.2),
            nn.Linear(512, 1024),
            lambda t: t.leaky_relu(0.2),
            nn.Linear(1024, 784),
            Tensor.tanh
        ]

    def __call__(self, x: Tensor) -> Tensor:
        # x = (BS, 128)
        return x.sequential(self.net)

class LinearDisc:
    def __init__(self) -> None:
        self.net = [
            nn.Linear(784, 1024),
            lambda t: t.leaky_relu(0.2),
            nn.Linear(1024, 512),
            lambda t: t.leaky_relu(0.2),
            nn.Linear(512, 256),
            lambda t: t.leaky_relu(0.2),
            nn.Linear(256, 1),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.net)

def make_batch(images: np.ndarray) -> Tensor:
    sample = np.random.randint(0, len(images), size=(BS))
    image_b = images[sample].reshape(-1, 28*28).astype(np.float32) / 127.5 - 1.0
    return Tensor(image_b, device=device)

def make_targets(bs: int, value: float) -> Tensor:
    # shape (bs, 1), values in [0, 1]
    return Tensor(np.full((bs, 1), value, dtype=np.float32), device=device)

def train_discriminator(discriminator, optimizer, data_real, data_fake) -> np.ndarray:
    # (512, 1)
    real_targets = make_targets(BS, 1.0)  # real -> 1
    fake_targets = make_targets(BS, 0.0)  # fake -> 0
    optimizer.zero_grad()
    # (512, 1)
    out_real = discriminator(data_real)   # logits
    out_fake = discriminator(data_fake)   # logits
    loss_real = out_real.bcewithlogitsloss(real_targets)
    loss_fake = out_fake.bcewithlogitsloss(fake_targets)
    d_loss = (loss_real + loss_fake) / 2.0
    d_loss.backward()
    optimizer.step()
    return d_loss.numpy()

def train_generator(discriminator, optimizer, data_fake):
    # non-saturating: push D(G(z)) to be real (target=1)
    real_like = make_targets(BS, 1.0)
    optimizer.zero_grad()
    out_fake = discriminator(data_fake)   # logits
    g_loss = out_fake.bcewithlogitsloss(real_like)
    g_loss.backward()
    optimizer.step()
    return g_loss.numpy()

if __name__ == "__main__":
    x_train_images, x_train_labels, x_test_images, x_test_labels = mnist(device=device)

    # data for training and validation
    images_real = np.vstack((np.asarray(x_train_images.numpy()), np.asarray(x_test_images.numpy())))
    ds_noise = Tensor.uniform((BS, 128), low=-1, high=1, device=device, requires_grad=False)

    # models and optimizer
    generator = LinearGen()
    discriminator = LinearDisc()

    # path to store results
    output_dir = Path(".").resolve() / "outputs"
    output_dir.mkdir(exist_ok=True)

    # optimizers
    optim_g = nn.optim.Adam(nn.utils.get_parameters(generator), lr=0.0002, betas=(0.5, 0.999))
    optim_d = nn.optim.Adam(nn.utils.get_parameters(discriminator), lr=0.0002, betas=(0.5, 0.999))

    # training loop
    for epoch in tqdm(range(EPOCHS), position=0, leave=True):
        loss_g, loss_d = 0.0, 0.0
        for _ in tqdm(range(STEPS), position=1, leave=False):
            # (512, 784)
            data_real = make_batch(images_real)
            # (512, 128)
            noise = Tensor.uniform((BS, 128), low=-1, high=1, device=device)
            # (512, 784)
            data_fake = generator(noise).detach()
            loss_d += train_discriminator(discriminator, optim_d, data_real, data_fake)
            noise = Tensor.uniform((BS, 128), low=-1, high=1, device=device)
            data_fake = generator(noise)
            loss_g += train_generator(discriminator, optim_g, data_fake)

        fake_images = generator(ds_noise).detach().numpy()
        fake_images = (fake_images.reshape(-1, 1, 28, 28) + 1) / 2  # 0 - 1 range.
        save_image(make_grid(torch.tensor(fake_images)[:64], nrow=8), output_dir / f"image_{epoch+1}.jpg")
        tqdm.write(f"Generator loss: {loss_g/STEPS}, Discriminator loss: {loss_d/STEPS}")

    print("Training Completed!")
