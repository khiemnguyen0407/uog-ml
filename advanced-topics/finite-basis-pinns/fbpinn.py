# %%  Import packages and module
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# %% Define Local Neural Networks
# Define Local Neural Networks

class MLP(nn.Module):

    def __init__(self, hidden_size=16):


        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)

# Subdomain stores not only the the coordinates that characterizes the subdomain
# but also the neural networks (MLPs) that are associated with this particular
# subdomain

class Subdomain:

    def __init__(self, xmin, xmax, hidden_size=16):
        self.xmin = xmin
        self.xmax = xmax

        self.model = MLP(hidden_size=hidden_size)

# This function normalizes x in the domain [xmin, xmax] to
# the reference domain [-1, 1]
def normalize(x, xmin, xmax):
    center = 0.5 * (xmin + xmax)
    half_width = 0.5 * (xmax - xmin)
    return (x - center)/half_width

# example
xx = torch.linspace(0, 0.6, 100)
xx_norm = normalize(xx, 0.0, 0.6)
plt.plot(xx.detach(), xx_norm)
# %%
# Local basis function
