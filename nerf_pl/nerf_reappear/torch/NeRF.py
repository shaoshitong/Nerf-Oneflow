import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27,
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

        self.reset_parameters(self)

    def reset_parameters(self,modules):
        for module in modules.modules():
            if isinstance(module,nn.Linear):
                nn.init.constant_(module.weight.data,0.1)
                if module.bias is not None:
                    nn.init.constant_(module.weight.data,0.01)

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        out = torch.cat([rgb, sigma], -1)

        return out

# model=NeRF()
# input=torch.linspace(0,10,90*2).view(2,90)
# output=model(input)
# print(input,output)
# tensor([[ 0.0000,  0.0559,  0.1117,  0.1676,  0.2235,  0.2793,  0.3352,  0.3911,
#           0.4469,  0.5028,  0.5587,  0.6145,  0.6704,  0.7263,  0.7821,  0.8380,
#           0.8939,  0.9497,  1.0056,  1.0615,  1.1173,  1.1732,  1.2291,  1.2849,
#           1.3408,  1.3966,  1.4525,  1.5084,  1.5642,  1.6201,  1.6760,  1.7318,
#           1.7877,  1.8436,  1.8994,  1.9553,  2.0112,  2.0670,  2.1229,  2.1788,
#           2.2346,  2.2905,  2.3464,  2.4022,  2.4581,  2.5140,  2.5698,  2.6257,
#           2.6816,  2.7374,  2.7933,  2.8492,  2.9050,  2.9609,  3.0168,  3.0726,
#           3.1285,  3.1844,  3.2402,  3.2961,  3.3520,  3.4078,  3.4637,  3.5196,
#           3.5754,  3.6313,  3.6872,  3.7430,  3.7989,  3.8547,  3.9106,  3.9665,
#           4.0223,  4.0782,  4.1341,  4.1899,  4.2458,  4.3017,  4.3575,  4.4134,
#           4.4693,  4.5251,  4.5810,  4.6369,  4.6927,  4.7486,  4.8045,  4.8603,
#           4.9162,  4.9721],
#         [ 5.0279,  5.0838,  5.1397,  5.1955,  5.2514,  5.3073,  5.3631,  5.4190,
#           5.4749,  5.5307,  5.5866,  5.6425,  5.6983,  5.7542,  5.8101,  5.8659,
#           5.9218,  5.9777,  6.0335,  6.0894,  6.1453,  6.2011,  6.2570,  6.3128,
#           6.3687,  6.4246,  6.4804,  6.5363,  6.5922,  6.6480,  6.7039,  6.7598,
#           6.8156,  6.8715,  6.9274,  6.9832,  7.0391,  7.0950,  7.1508,  7.2067,
#           7.2626,  7.3184,  7.3743,  7.4302,  7.4860,  7.5419,  7.5978,  7.6536,
#           7.7095,  7.7654,  7.8212,  7.8771,  7.9330,  7.9888,  8.0447,  8.1006,
#           8.1564,  8.2123,  8.2682,  8.3240,  8.3799,  8.4358,  8.4916,  8.5475,
#           8.6034,  8.6592,  8.7151,  8.7709,  8.8268,  8.8827,  8.9385,  8.9944,
#           9.0503,  9.1061,  9.1620,  9.2179,  9.2737,  9.3296,  9.3855,  9.4413,
#           9.4972,  9.5531,  9.6089,  9.6648,  9.7207,  9.7765,  9.8324,  9.8883,
#           9.9441, 10.0000]]) tensor([[1.0000e+00, 1.0000e+00, 1.0000e+00, 2.0497e+03],
#         [1.0000e+00, 1.0000e+00, 1.0000e+00, 8.0290e+03]],
#        grad_fn=<CatBackward0>)