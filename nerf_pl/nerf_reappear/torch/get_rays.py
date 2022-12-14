import torch

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

# x1=torch.linspace(0,10,10*10*3).view(10,10,3)
# x2=torch.linspace(0,10,3*4).view(3,4)
# out1,out2=get_rays(x1,x2)
# print(out1,out2)
# """
# tensor([[ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000],
#         [ 2.7273,  6.3636, 10.0000]]) tensor([[0.1471, 0.5002, 0.8533],
#         [0.1102, 0.4880, 0.8658],
#         [0.1045, 0.4861, 0.8677],
#         [0.1022, 0.4853, 0.8684],
#         [0.1009, 0.4848, 0.8688],
#         [0.1001, 0.4846, 0.8690],
#         [0.0996, 0.4844, 0.8692],
#         [0.0992, 0.4842, 0.8693],
#         [0.0989, 0.4841, 0.8694],
#         [0.0986, 0.4841, 0.8695],
#         [0.0985, 0.4840, 0.8695],
#         [0.0983, 0.4839, 0.8696],
#         [0.0982, 0.4839, 0.8696],
#         [0.0981, 0.4838, 0.8696],
#         [0.0980, 0.4838, 0.8697],
#         [0.0979, 0.4838, 0.8697],
#         [0.0978, 0.4838, 0.8697],
#         [0.0977, 0.4837, 0.8697],
#         [0.0977, 0.4837, 0.8698],
#         [0.0976, 0.4837, 0.8698],
#         [0.0976, 0.4837, 0.8698],
#         [0.0975, 0.4837, 0.8698],
#         [0.0975, 0.4837, 0.8698],
#         [0.0975, 0.4836, 0.8698],
#         [0.0974, 0.4836, 0.8698],
#         [0.0974, 0.4836, 0.8698],
#         [0.0974, 0.4836, 0.8698],
#         [0.0973, 0.4836, 0.8699],
#         [0.0973, 0.4836, 0.8699],
#         [0.0973, 0.4836, 0.8699],
#         [0.0973, 0.4836, 0.8699],
#         [0.0973, 0.4836, 0.8699],
#         [0.0972, 0.4836, 0.8699],
#         [0.0972, 0.4836, 0.8699],
#         [0.0972, 0.4836, 0.8699],
#         [0.0972, 0.4835, 0.8699],
#         [0.0972, 0.4835, 0.8699],
#         [0.0972, 0.4835, 0.8699],
#         [0.0972, 0.4835, 0.8699],
#         [0.0971, 0.4835, 0.8699],
#         [0.0971, 0.4835, 0.8699],
#         [0.0971, 0.4835, 0.8699],
#         [0.0971, 0.4835, 0.8699],
#         [0.0971, 0.4835, 0.8699],
#         [0.0971, 0.4835, 0.8699],
#         [0.0971, 0.4835, 0.8699],
#         [0.0971, 0.4835, 0.8699],
#         [0.0971, 0.4835, 0.8699],
#         [0.0971, 0.4835, 0.8699],
#         [0.0970, 0.4835, 0.8699],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0970, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4835, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700],
#         [0.0969, 0.4834, 0.8700]])
#
# """
