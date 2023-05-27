import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from sympy.tensor import tensor
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.nn.functional import conv2d, max_pool2d, cross_entropy

plt.rc("figure", dpi=100)

batch_size = 100

# transform images into normalized tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_dataset = datasets.MNIST(
    "./",
    download=True,
    train=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    "./",
    download=True,
    train=False,
    transform=transform,
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)


def init_weights(shape):
    # Kaiming He initialization (a good initialization is important)
    # https://arxiv.org/abs/1502.01852
    std = np.sqrt(2. / shape[0])
    w = torch.randn(size=shape) * std
    w.requires_grad = True
    return w


def rectify(x):
    # Rectified Linear Unit (ReLU)
    return torch.max(torch.zeros_like(x), x)


# 3
def PRelu(X, a):
    return torch.where(X <= 0, a * X, X)


class RMSprop(optim.Optimizer):
    """
    This is a reduced version of the PyTorch internal RMSprop optimizer
    It serves here as an example
    """

    def __init__(self, params, lr=1e-3, alpha=0.5, eps=1e-8):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(RMSprop, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # state initialization
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                # update running averages
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                avg = square_avg.sqrt().add_(group['eps'])

                # gradient update
                p.data.addcdiv_(grad, avg, value=-group['lr'])


# 2
def dropout(X, p_drop=0.5):
    mask = np.random.binomial(1, 1 - p_drop, size=X.shape) / (1 - p_drop)
    return X * torch.tensor(mask, dtype=torch.float32)


def dropout_model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden, a):
    # Dropout at the input layer
    X = dropout(X, p_drop=p_drop_input)
    X = X.to(w_h.dtype)  # Convert X to the same data type as w_h
    h = X @ w_h
    # a1 = torch.nn.Parameter(torch.zeros_like(h), requires_grad=True)
    # Forward pass through the first hidden layer
    h = PRelu(h, a)
    # Dropout at the first hidden layer
    h = dropout(h, p_drop=p_drop_hidden)
    # Forward pass through the second hidden layer
    h2 = h @ w_h2
    # a2 = torch.nn.Parameter(torch.zeros_like(h2), requires_grad=True)
    h2 = PRelu(h2, a)
    # Dropout at the second hidden layer
    h2 = dropout(h2, p_drop=p_drop_hidden)
    # Forward pass through the output layer
    output = h2 @ w_o

    return output


def main():
    # initialize weights

    # input shape is (B, 784)
    w_h = init_weights((784, 625))
    # hidden layer with 625 neurons
    w_h2 = init_weights((625, 625))
    # hidden layer with 625 neurons
    w_o = init_weights((625, 10))
    a = init_weights((100, 625))
    # output shape is (B, 10)

    optimizer = RMSprop(params=[w_h, w_h2, w_o, a])

    n_epochs = 100

    train_loss = []
    test_loss = []

    # put this into a training loop over 100 epochs
    for epoch in range(n_epochs + 1):
        train_loss_this_epoch = []
        for idx, batch in enumerate(train_dataloader):
            x, y = batch

            # our model requires flattened input
            x = x.reshape(batch_size, 784)
            # feed input through model
            # noise_py_x = model(x, w_h, w_h2, w_o)
            # noise_py_x = dropout_model(x, w_h, w_h2, w_o, 0.5, 0.5)
            noise_py_x = dropout_model(x, w_h, w_h2, w_o, 0.5, 0.5, a)

            # reset the gradient
            optimizer.zero_grad()

            # the cross-entropy loss function already contains the softmax
            loss = cross_entropy(noise_py_x, y, reduction="mean")

            train_loss_this_epoch.append(float(loss))

            # compute the gradient
            loss.backward()
            # update weights
            optimizer.step()

        train_loss.append(np.mean(train_loss_this_epoch))

        # test periodically
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")
            print(f"Mean Train Loss: {train_loss[-1]:.2e}")
            test_loss_this_epoch = []

            # no need to compute gradients for validation
            with torch.no_grad():
                for idx, batch in enumerate(test_dataloader):
                    x, y = batch
                    x = x.reshape(batch_size, 784)
                    noise_py_x = dropout_model(x, w_h, w_h2, w_o, 0.5, 0.5,a )

                    loss = cross_entropy(noise_py_x, y, reduction="mean")
                    test_loss_this_epoch.append(float(loss))

            test_loss.append(np.mean(test_loss_this_epoch))

            print(f"Mean Test Loss:  {test_loss[-1]:.2e}")

    plt.plot(np.arange(n_epochs + 1), train_loss, label="Train")
    plt.plot(np.arange(1, n_epochs + 2, 10), test_loss, label="Test")
    plt.title("Train and Test Loss over Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


if __name__ == '__main__':
    main()
