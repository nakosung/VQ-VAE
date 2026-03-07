import torch
import torch.nn as nn
import math


class VectorQuantize(torch.autograd.Function):
    """Straight-through estimator for vector quantization."""

    @staticmethod
    def forward(ctx, z, codebook):
        # z: (B*H*W, z_dim), codebook: (k_dim, z_dim)
        # L2 distance: find nearest codebook entry
        dist = ((z[:, None] - codebook[None, :]) ** 2).sum(2)
        indices = dist.min(1)[1]
        z_q = codebook[indices]

        ctx.save_for_backward(z, z_q, codebook, indices)
        # Straight-through: copy gradients from z_q to z
        z_q_st = z + (z_q - z).detach()
        return z_q_st, z_q, indices

    @staticmethod
    def backward(ctx, grad_z_q_st, grad_z_q, grad_indices):
        z, z_q, codebook, indices = ctx.saved_tensors
        # Gradient for z passes straight through
        grad_z = grad_z_q_st.clone()
        # No gradient for codebook here; losses handle codebook updates
        return grad_z, None


vector_quantize = VectorQuantize.apply


class Generator(nn.Module):
    """Generator. Vector Quantised Variational Auto-Encoder."""

    def __init__(self, image_size=64, z_dim=256, conv_dim=64, code_dim=16, k_dim=256):
        super().__init__()

        self.k_dim = k_dim
        self.z_dim = z_dim
        self.code_dim = code_dim

        self.codebook = nn.Embedding(k_dim, z_dim)

        # Encoder (increasing #filter linearly)
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.ReLU())

        repeat_num = int(math.log2(image_size / code_dim))
        curr_dim = conv_dim
        for i in range(repeat_num):
            layers.append(nn.Conv2d(curr_dim, conv_dim * (i + 2), kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(conv_dim * (i + 2)))
            layers.append(nn.ReLU())
            curr_dim = conv_dim * (i + 2)

        # Now we have (code_dim, code_dim, curr_dim)
        layers.append(nn.Conv2d(curr_dim, z_dim, kernel_size=1))

        # (code_dim, code_dim, z_dim)
        self.encoder = nn.Sequential(*layers)

        # Decoder
        layers = []
        layers.append(nn.ConvTranspose2d(z_dim, curr_dim, kernel_size=1))
        layers.append(nn.BatchNorm2d(curr_dim))
        layers.append(nn.ReLU())

        for i in reversed(range(repeat_num)):
            layers.append(nn.ConvTranspose2d(curr_dim, conv_dim * (i + 1), kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(conv_dim * (i + 1)))
            layers.append(nn.ReLU())
            curr_dim = conv_dim * (i + 1)

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)

        self._init_codebook()

    def _init_codebook(self):
        initrange = 1.0 / self.k_dim
        self.codebook.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        h = self.encoder(x)
        sz = h.size()

        # BCHW -> BHWC -> (B*H*W, C)
        h_flat = h.permute(0, 2, 3, 1).contiguous().view(-1, self.z_dim)

        W = self.codebook.weight

        # Vector quantization with straight-through estimator
        z_q_st, z_q, indices = vector_quantize(h_flat, W)

        # Reshape back to BCHW
        h_q = z_q_st.view(sz[0], sz[2], sz[3], sz[1]).permute(0, 3, 1, 2)

        # VQ losses
        loss_codebook = ((h_flat.detach() - z_q) ** 2).sum(1).mean()
        loss_commit = ((h_flat - z_q.detach()) ** 2).sum(1).mean()

        return self.decoder(h_q), loss_codebook, loss_commit

    def decode(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.decoder(z)
