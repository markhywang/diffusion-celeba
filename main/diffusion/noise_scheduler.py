import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T, BETA_START, BETA_END = 300, 1e-4, 2e-2


def linear_beta_transform(timesteps, start, end):
    """
    Return beta scheduler used in image noising process
    """
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    
    # Return a shape that matches the image shape to prevent shape errors
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion(x_0, device, t, sqrt_alphas_cumprod, 
                      sqrt_one_minus_alphas_cumprod):
    """
    Adds noise to a particular image with respect to a timestamp.
    """
    noise = torch.randn_like(x_0).to(device)
    
    # Square root of alpha values at timestamp t
    sqrt_alphas_cumprod_t = get_index_from_list(
        sqrt_alphas_cumprod, 
        t, 
        x_0.shape
    )
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod,
        t,
        x_0.shape
    )
    
    # Mean and variance
    mean_t = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
    variance_t = sqrt_one_minus_alphas_cumprod_t.to(device) * noise
    
    return mean_t + variance_t, noise


if __name__ == '__main__':
    """Sample pre-processing"""
    betas = linear_beta_transform(T, BETA_START, BETA_END)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
