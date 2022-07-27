import torch

x = torch.linspace(1, 11, 10)
y = torch.linspace(2, 12, 10)
x=x.unsqueeze(-1).expand(x.size(0),10).contiguous()
y=y.unsqueeze(-1).expand(y.size(0),100).contiguous()
print(x.shape,y.shape)
res1 = torch.searchsorted(x, y) # x排序后，y在x中搜寻每个值出现的位置
print(res1)
