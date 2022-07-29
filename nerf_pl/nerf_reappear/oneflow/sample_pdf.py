import oneflow as flow
import oneflow.nn as nn


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / flow.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = flow.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = flow.cat([flow.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = flow.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = flow.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = flow.searchsorted(cdf, u, right=True)
    below = flow.clamp(inds - 1, 0, 1e6)
    above = flow.clamp(inds, -1e6, N_samples_)

    inds_sampled = flow.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = flow.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = flow.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples

#
# if __name__=="__main__":
#     N_rays,N_samples=10,20
#     bins=flow.linspace(0,10,10*(20+1)).view(10,-1)
#     weights=flow.linspace(0,10,10*20).view(10,-1).softmax(-1)
#     samples=sample_pdf(bins,weights,20,det=True)
#     print(samples)
#     """
#     tensor([[ 0.0000,  0.0828,  0.1592,  0.2300,  0.2958,  0.3573,  0.4151,  0.4697,
#           0.5214,  0.5703,  0.6168,  0.6611,  0.7034,  0.7439,  0.7829,  0.8204,
#           0.8564,  0.8909,  0.9244,  0.9569],
#         [ 1.0048,  1.0876,  1.1640,  1.2348,  1.3006,  1.3620,  1.4199,  1.4745,
#           1.5261,  1.5751,  1.6216,  1.6659,  1.7082,  1.7487,  1.7876,  1.8252,
#           1.8612,  1.8957,  1.9291,  1.9617],
#         [ 2.0096,  2.0924,  2.1687,  2.2395,  2.3054,  2.3668,  2.4247,  2.4793,
#           2.5309,  2.5799,  2.6264,  2.6707,  2.7130,  2.7535,  2.7924,  2.8299,
#           2.8660,  2.9005,  2.9339,  2.9665],
#         [ 3.0144,  3.0972,  3.1735,  3.2443,  3.3102,  3.3716,  3.4295,  3.4841,
#           3.5357,  3.5846,  3.6311,  3.6754,  3.7177,  3.7583,  3.7972,  3.8347,
#           3.8708,  3.9052,  3.9387,  3.9713],
#         [ 4.0191,  4.1020,  4.1783,  4.2491,  4.3149,  4.3764,  4.4343,  4.4889,
#           4.5405,  4.5894,  4.6359,  4.6802,  4.7225,  4.7631,  4.8020,  4.8395,
#           4.8755,  4.9100,  4.9435,  4.9761],
#         [ 5.0239,  5.1067,  5.1831,  5.2539,  5.3197,  5.3812,  5.4390,  5.4936,
#           5.5453,  5.5942,  5.6407,  5.6850,  5.7273,  5.7678,  5.8068,  5.8443,
#           5.8803,  5.9148,  5.9483,  5.9809],
#         [ 6.0287,  6.1115,  6.1879,  6.2587,  6.3245,  6.3860,  6.4438,  6.4984,
#           6.5501,  6.5990,  6.6455,  6.6898,  6.7321,  6.7726,  6.8116,  6.8491,
#           6.8851,  6.9196,  6.9531,  6.9856],
#         [ 7.0335,  7.1163,  7.1927,  7.2635,  7.3293,  7.3907,  7.4486,  7.5032,
#           7.5548,  7.6038,  7.6503,  7.6946,  7.7369,  7.7774,  7.8164,  7.8539,
#           7.8899,  7.9244,  7.9579,  7.9904],
#         [ 8.0383,  8.1211,  8.1975,  8.2682,  8.3341,  8.3955,  8.4534,  8.5080,
#           8.5596,  8.6086,  8.6551,  8.6994,  8.7417,  8.7822,  8.8211,  8.8587,
#           8.8947,  8.9292,  8.9626,  8.9952],
#         [ 9.0431,  9.1259,  9.2022,  9.2730,  9.3389,  9.4003,  9.4582,  9.5128,
#           9.5644,  9.6133,  9.6598,  9.7041,  9.7465,  9.7870,  9.8259,  9.8634,
#           9.8995,  9.9340,  9.9674, 10.0000]], dtype=oneflow.float32)
#     """