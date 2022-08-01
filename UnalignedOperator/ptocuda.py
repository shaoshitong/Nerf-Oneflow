import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self,N_freqs=10):
        super(Embedding, self).__init__()
        self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
    def forward(self,x, *args, **kwargs):
        out = [x]
        print(self.freq_bands.device)
        for freq in self.freq_bands:
            print(freq.device)
            out += [(freq*x)]
        return torch.cat(out, -1)

model=Embedding().cuda()
input=torch.ones(1,10).cuda()
output=model(input)
"""
/root/anaconda3/envs/torch/bin/python /home/sst/product/Nerf-Oneflow/UnalignedOperator/ptocuda.py

Process finished with exit code 0
"""
