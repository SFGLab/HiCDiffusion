import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch import nn
import math
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
from tqdm import tqdm
# def symmetricND(size: int, dims: int) -> torch.Tensor:
#     data = torch.randn(*[size] * dims)
#     return functools.reduce(
#         operator.add,
#         (
#             torch.permute(data, permutation)
#             for permutation in itertools.permutations(range(dims))
#         ),
#     )

# def random_symetric(shape, device):
#     tensor_list = []
#     for i in range(0, shape[0]):
#         tensor = symmetricND(shape[2], 2).to(device).view(1, 1, shape[2], shape[3])
#         tensor_list.append(tensor)
#     tensor = torch.cat(tensor_list)
#     return tensor

# def randn_like_symetric(img):
#     return random_symetric(img.shape, img.device)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class UnetConditional(Unet):
    
    def __init__(self, dim, dim_mults, flash_attn, channels):
        super().__init__(dim=dim, dim_mults=dim_mults, flash_attn=flash_attn, channels=channels)
        self.init_conv = nn.Identity() # we will handle initial conv ourselves :)
        
        self.init_conv_x = nn.Conv2d(2, dim, 7, padding = 3)
        # self.init_conv_x_cond = nn.Conv2d(512, dim, 7, padding = 3)

    def forward(self, x, time, x_self_cond):
        x = torch.cat((x, x_self_cond), 1)
        x = self.init_conv_x(x)
        x = super().forward(x, time)
        return x

class GaussianDiffusionConditional(GaussianDiffusion):
    def __init__(self, model, image_size, timesteps, sampling_timesteps):
        super().__init__(model, image_size=image_size, timesteps=timesteps, sampling_timesteps=sampling_timesteps, auto_normalize=False)

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    
    @torch.inference_mode()
    def p_sample_loop(self, shape, cond, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    def norm(self, tensor):
        norm_min = 0
        norm_max = 10
        return 2*(tensor-norm_min)/(norm_max-norm_min)-1
    def denorm(self, tensor):
        norm_min = 0
        norm_max = 10
        return (tensor+1)/2*(norm_max-norm_min)+norm_min
    
    def p_losses(self, x_start, t, x_self_cond = None, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape
        
        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start-x_self_cond, t = t, noise = noise)

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    
    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False, x_self_cond=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), x_self_cond, return_all_timesteps = return_all_timesteps)