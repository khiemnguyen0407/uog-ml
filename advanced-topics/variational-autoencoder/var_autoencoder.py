import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # --- ENCODER ---
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Linear layers to output the mean (mu) and log-variance (logvar) of the latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- DECODER ---
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim) # Output matches input dimension

    def encode(self, x):
        """Maps the input x to the parameters of the latent distribution (mu and logvar)."""
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Performs the reparameterization trick for differentiable sampling."""
        # Calculate standard deviation (sigma)
        std = torch.exp(0.5 * logvar)
        # Sample random noise epsilon from a standard normal distribution
        eps = torch.randn_like(std)
        # Calculate latent vector z = mu + sigma * epsilon
        z = mu + eps * std
        return z

    def decode(self, z):
        """Maps the latent vector z back to the data space (reconstruction)."""
        h = F.relu(self.fc3(z))
        # Use sigmoid for binary data like MNIST (pixels between 0 and 1)
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        """The full pass: encode, reparameterize, and decode."""
        mu, logvar = self.encode(x.view(-1, 784)) # Flatten input for the linear layers
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

# --- VAE LOSS FUNCTION ---
def loss_function(recon_x, x, mu, logvar):
    """
    Computes the total VAE loss (Reconstruction Loss + KL Divergence).
    """
    # 1. Reconstruction Loss (Binary Cross-Entropy for MNIST)
    # Binary Cross-Entropy (BCE) measures the difference between the
    # reconstructed image (recon_x) and the original image (x).
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # 2. KL Divergence (Regularization Loss)
    # Measures the divergence between the learned latent distribution q(z|x)
    # and the prior distribution p(z) (Standard Normal N(0, 1)).
    # The formula is: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD