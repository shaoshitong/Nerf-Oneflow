import oneflow as flow
import oneflow.nn as nn

class Embedding(nn.Module):
    def __init__(self,N_freqs=10):
        super(Embedding, self).__init__()
        self.freq_bands = 2**flow.linspace(0, N_freqs-1, N_freqs)
    def forward(self,x, *args, **kwargs):
        out = [x]
        for freq in self.freq_bands:
            out += [(freq*x)]
        return flow.cat(out, -1)

model=Embedding().cuda()
input=flow.ones(1,10).cuda()
output=model(input)
"""
/root/anaconda3/envs/torch/bin/python /home/sst/product/Nerf-Oneflow/UnalignedOperator/tocuda.py
libibverbs not available, ibv_fork_init skipped
Traceback (most recent call last):
  File "/home/sst/product/Nerf-Oneflow/UnalignedOperator/tocuda.py", line 16, in <module>
    output=model(input)
  File "/root/anaconda3/envs/torch/lib/python3.7/site-packages/oneflow/nn/module.py", line 115, in __call__
    res = self.forward(*args, **kwargs)
  File "/home/sst/product/Nerf-Oneflow/UnalignedOperator/tocuda.py", line 11, in forward
    out += [(freq*x)]
RuntimeError: Check failed: *default_device == *input_device Expected all tensors to be on the same device, but found at least two devices, cpu:0 (positional 0) and cuda:0 (positional 1)!

Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
"""