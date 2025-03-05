import torch
from torch import nn
import math


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch),
            nn.ReLU()
        )
        
        self.conv1 = nn.Sequential(
            # If up=True, then due to residual connections, input=2*in_ch
            nn.Conv2d((up + 1) * in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # Additional conv for further spatial learning but preserves dimensions
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        
        if up: # If upscaling, then perform de-convolution
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else: # If downscaling, then perform convolution
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        
    def forward(self, x, t):
        # First Conv
        h = self.conv1(x)

        # Time embedding
        time_emb = self.time_mlp(t)
        time_emb = time_emb.view(*time_emb.shape, 1, 1) # Add 2 dimension to match 'h'

        # Add time embedding to represent position
        h = h + time_emb

        # Second Conv
        h = self.conv2(h)

        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(1e4) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class U_Net(nn.Module):
    """
    A simple U-Net architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        conv_channels = (64, 128, 256, 512, 1024)
        n = len(conv_channels)
        out_dim = 3 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, conv_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([
            Block(conv_channels[i], conv_channels[i+1], time_emb_dim) for i in range(n-1)
        ])
        
        # Upsample (reverses process for channels)
        self.ups = nn.ModuleList([
            Block(conv_channels[i], conv_channels[i-1], time_emb_dim, up=True) for i in range(n-1, 0, -1)
        ])

        self.output = nn.Conv2d(conv_channels[0], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        
        # Initial conv
        x = self.conv0(x)

        residual_inputs = []
        
        for down_sample_block in self.downs:
            x = down_sample_block(x, t)
            residual_inputs.append(x)
        
        for up_sample_block in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1) # Residual connections    
            x = up_sample_block(x, t)

        return self.output(x)