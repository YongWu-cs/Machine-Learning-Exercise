import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
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


def dropout(X, p_drop=0.5):
    mask = np.random.binomial(1, 1 - p_drop, size=X.shape) / (1 - p_drop)
    return X * torch.tensor(mask, dtype=torch.float32)


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


# def model(x, w_h1, w_h2, w_h3, w_o):
#     convolutional_layer = rectify(conv2d(x, w_h1))  # reduces (2,2) window to 1 pixel
#     subsampling_layer = max_pool2d(convolutional_layer, (2, 2))
#     out_layer = dropout(subsampling_layer, p_drop=0.5)
#     # print(2)
#     convolutional_layer = rectify(conv2d(out_layer, w_h2))  # reduces (2,2) window to 1 pixel
#     subsampling_layer = max_pool2d(convolutional_layer, (2, 2))
#     out_layer = dropout(subsampling_layer, p_drop=0.5)
#     # print(2)
#     convolutional_layer = rectify(conv2d(out_layer, w_h3))  # reduces (2,2) window to 1 pixel
#     subsampling_layer = max_pool2d(convolutional_layer, (2, 2))
#     out_layer = dropout(subsampling_layer, p_drop=0.5)
#     print(2)
#     pre_softmax = out_layer.view(out_layer.size(0), -1)
#     return pre_softmax
def model(x, w_h1, w_h2, w_o):
    convolutional_layer = rectify(conv2d(x, w_h1))  # reduces (2,2) window to 1 pixel
    subsampling_layer = max_pool2d(convolutional_layer, (2, 2))
    h1 = dropout(subsampling_layer, p_drop=0.5)

    convolutional_layer = rectify(conv2d(h1, w_h2))  # reduces (2,2) window to 1 pixel
    subsampling_layer = max_pool2d(convolutional_layer, (2, 2))
    h2 = dropout(subsampling_layer, p_drop=0.5)

    convolutional_layer = rectify(conv2d(h2, w_o))  # reduces (2,2) window to 1 pixel
    subsampling_layer = max_pool2d(convolutional_layer, (2, 2))
    h3 = dropout(subsampling_layer, p_drop=0.5)

    pre_softmax = h3.view(h3.size(0), -1)
    return pre_softmax


def main():
    # initialize weights
    w_h1 = init_weights((32, 1, 5, 5))
    w_h2 = init_weights((64, 32, 5, 5))
    w_o = init_weights((128, 64, 2, 2))


    optimizer = RMSprop(params=[w_h1, w_h2, w_o])

    n_epochs = 100

    train_loss = []
    test_loss = []

    # put this into a training loop over 100 epochs
    for epoch in range(n_epochs + 1):
        train_loss_this_epoch = []
        for idx, batch in enumerate(train_dataloader):
            x, y = batch

            # our model requires flattened input
            x = x.reshape(-1, 1, 28, 28)
            y = y.reshape(-1, 1, 28, 28)
            # feed input through model
            noise_py_x = model(x, w_h1, w_h2, w_o)

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
                    x = x.reshape(-1, 1, 28, 28)
                    y = y.reshape(-1, 1, 28, 28)
                    noise_py_x = model(x, w_h1, w_h2, w_o)

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
