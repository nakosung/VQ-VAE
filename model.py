import torch 
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable

class Generator(nn.Module):
    """Generator. Vector Quantised Variational Auto-Encoder."""
    def __init__(self, image_size=64, z_dim=256, conv_dim=64, code_size=16, k_dim=256):
        super(Generator, self).__init__()

        self.k_dim = k_dim
        self.z_dim = z_dim
        self.code_size = code_size
        
        self.dict = nn.Embedding(k_dim, z_dim)
        
        # Encoder (increasing #filter linearly)
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.ReLU())
        
        repeat_num = int(math.log2(image_size / code_size))
        curr_dim = conv_dim
        for i in range(repeat_num):
            layers.append(nn.Conv2d(curr_dim, conv_dim * (i+2), kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(conv_dim * (i+2)))
            layers.append(nn.ReLU())
            curr_dim = conv_dim * (i+2)

        # Now we have (code_size,code_size,curr_dim)
        layers.append(nn.Conv2d(curr_dim, z_dim, kernel_size=1))

        # (code_size,code_size,z_dim)
        self.encoder = nn.Sequential(*layers)
        
        # Decoder (320 - 256 - 192 - 128 - 64)
        layers = []
        layers.append(nn.ConvTranspose2d(z_dim, curr_dim, kernel_size=1))
        layers.append(nn.BatchNorm2d(curr_dim))
        layers.append(nn.ReLU())
                
        for i in reversed(range(repeat_num)):
            layers.append(nn.ConvTranspose2d(curr_dim , conv_dim * (i+1), kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(conv_dim * (i+1)))
            layers.append(nn.ReLU())
            curr_dim = conv_dim * (i+1)
        
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)

        self.init_weights()        
        
    def init_weights(self):
        initrange = 1.0 / self.k_dim
        self.dict.weight.data.uniform_(-initrange, initrange)        
            
    def forward(self, x):
        h = self.encoder(x)                             # (?, z_dim*2, 1, 1)

        sz = h.size()
        
        # BCWH -> BWHC
        org_h = h
        h = h.permute(0,2,3,1)
        h = h.contiguous()
        Z = h.view(-1,self.z_dim)

        W = self.dict.weight

        def L2_dist(a,b):
            return ((a - b) ** 2)
        
        # Sample nearest embedding
        j = L2_dist(Z[:,None],W[None,:]).sum(2).min(1)[1]
        W_j = W[j]

        # Stop gradients
        Z_sg = Z.detach()
        W_j_sg = W_j.detach()

        # BWHC -> BCWH
        h = W_j.view(sz[0],sz[2],sz[3],sz[1])
        h = h.permute(0,3,1,2)

        def hook(grad):
            nonlocal org_h
            self.saved_grad = grad
            self.saved_h = org_h
            return grad

        h.register_hook(hook)
        
        # losses
        return self.decoder(h), L2_dist(Z,W_j_sg).sum(1).mean(), L2_dist(Z_sg,W_j).sum(1).mean()

    # back propagation for encoder
    def bwd(self):
        self.saved_h.backward(self.saved_grad)
        
    def decode(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.decoder(z)
    
