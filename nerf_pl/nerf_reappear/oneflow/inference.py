import oneflow as flow
import oneflow.nn as nn
import torch
from nerf_reappear.oneflow.NeRF import NeRF
from nerf_reappear.oneflow.Embedding import Embedding


def inference(N_rays, model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):
    # TODO: INITIAL SETTINGS
    N_samples = 64
    use_disp = False
    perturb = 0
    noise_std = 1
    N_importance = 0
    chunk = 1024 * 32
    white_back = False
    test_time = False
    """
    Helper function that performs model inference.

    Inputs:
        N_rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        model: NeRF model (coarse or fine)
        embedding_xyz: embedding module for xyz
        xyz_: (N_rays, N_samples_, 3) sampled positions
              N_samples_ is the number of sampled points in each ray;
                         = N_samples for coarse model
                         = N_samples+N_importance for fine model
        dir_: (N_rays, 3) ray directions
        dir_embedded: (N_rays, embed_dir_channels) embedded directions
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        weights_only: do inference on sigma only or not

    Outputs:
        if weights_only:
            weights: (N_rays, N_samples_): weights of each sample
        else:
            rgb_final: (N_rays, 3) the final rgb image
            depth_final: (N_rays) depth map
            weights: (N_rays, N_samples_): weights of each sample
    """
    N_samples_ = xyz_.shape[1]
    # Embed directions
    xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
    if not weights_only:
        dir_embedded = flow.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
        # (N_rays*N_samples_, embed_dir_channels)

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    out_chunks = []
    for i in range(0, B, chunk):
        # Embed positions by chunk
        xyz_embedded = embedding_xyz(xyz_[i:i + chunk])
        if not weights_only:
            xyzdir_embedded = flow.cat([xyz_embedded,
                                         dir_embedded[i:i + chunk]], 1)
        else:
            xyzdir_embedded = xyz_embedded
        out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]

    out = flow.cat(out_chunks, 0)
    if weights_only:
        sigmas = out.view(N_rays, N_samples_)
    else:
        rgbsigma = out.view(N_rays, N_samples_, 4)
        rgbs = rgbsigma[..., :3]  # (N_rays, N_samples_, 3)
        sigmas = rgbsigma[..., 3]  # (N_rays, N_samples_)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    delta_inf = 1e10 * flow.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
    deltas = flow.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * flow.norm(dir_.unsqueeze(1), dim=-1)

    noise = flow.randn(sigmas.shape, device=sigmas.device) * noise_std

    # compute alpha by the formula (3)
    alphas = 1 - flow.exp(-deltas * flow.relu(sigmas + noise))  # (N_rays, N_samples_)
    alphas_shifted = \
        flow.cat([flow.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
    weights = \
        alphas * flow.cumprod(alphas_shifted, -1)[:, :-1]  # (N_rays, N_samples_)
    weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    if weights_only:
        return weights

    # compute final weighted outputs
    rgb_final = flow.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
    depth_final = flow.sum(weights * z_vals, -1)  # (N_rays)

    if white_back:
        rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

    return rgb_final, depth_final, weights

# if __name__ == '__main__':
#
#     model=inference
#     input=flow.linspace(1,10,8*(3+3+2)).view(8,-1).cuda()
#     nerf=[NeRF().cuda()]
#     embeddings = [Embedding(3,10).cuda(), Embedding(3,4).cuda()]
#     rays=flow.linspace(0,100,1024*8).view(1024,-1).cuda() # flow.Size([160000, 8]) rays # TODO: --dataset_name blender
#     rays=rays[0:1024].cuda()
#     model_coarse = nerf[0]
#     embedding_xyz = embeddings[0]
#     embedding_dir = embeddings[1]
#     N_samples=64
#     use_disp=False
#     perturb=1.0
#     # Decompose the inputs
#     N_rays = rays.shape[0]
#     rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
#     near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
#
#     # Embed direction
#     dir_embedded = embedding_dir(rays_d)  # (N_rays, embed_dir_channels)
#
#     # Sample depth points
#     z_steps = flow.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
#     if not use_disp:  # use linear sampling in depth space
#         z_vals = near * (1 - z_steps) + far * z_steps
#     else:  # use linear sampling in disparity space
#         z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)
#
#     z_vals = z_vals.expand(N_rays, N_samples)
#
#     if perturb > 0:  # perturb sampling depths (z_vals)
#         z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
#         # get intervals between samples
#         upper = flow.cat([z_vals_mid, z_vals[:, -1:]], -1)
#         lower = flow.cat([z_vals[:, :1], z_vals_mid], -1)
#
#         perturb_rand = perturb * flow.rand(z_vals.shape, device=rays.device)
#         z_vals = lower + (upper - lower) * perturb_rand
#
#     xyz_coarse_sampled = rays_o.unsqueeze(1) + \
#                          rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)
#
#     output=model(rays.shape[0],model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
#                           dir_embedded, z_vals, weights_only=False)
#     print(output)
#     """
#     能跑通但并未测试是否对齐
#     """