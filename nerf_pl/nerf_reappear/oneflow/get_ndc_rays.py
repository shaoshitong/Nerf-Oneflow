import oneflow as flow

def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (W / (2. * focal)) * ox_oz
    o1 = -1. / (H / (2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = flow.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = flow.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d

# x1=flow.linspace(0,10,5)
# x2=flow.linspace(0,10,5*3).view(5,3)
# x3=flow.linspace(0,10,5*3).view(5,3)
#
# rays_o,rays_d=get_ndc_rays(10,10,10,x1,x2,x3)
# print(rays_o,rays_d)
# tensor([[    nan,     nan,     nan],
#         [-1.2000, -1.6000, -1.0000],
#         [-1.5000, -1.7500, -1.0000],
#         [-1.6364, -1.8182, -1.0000],
#         [-1.7143, -1.8571, -1.0000]], dtype=oneflow.float32) tensor([[        nan,         nan,         nan],
#         [-1.1921e-07,  1.1921e-07,  2.0000e+00],
#         [ 1.1921e-07, -0.0000e+00,  2.0000e+00],
#         [-1.1921e-07,  1.1921e-07,  2.0000e+00],
#         [-0.0000e+00, -0.0000e+00,  2.0000e+00]], dtype=oneflow.float32)
