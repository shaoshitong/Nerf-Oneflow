import oneflow as flow

x=flow.linspace(0,10,100)
y=flow.clamp_max(x,10)
print(y)
"""
/root/anaconda3/envs/torch/bin/python /home/sst/product/Nerf-Oneflow/UnalignedOperator/oclamp.py
libibverbs not available, ibv_fork_init skipped
Traceback (most recent call last):
  File "/home/sst/product/Nerf-Oneflow/UnalignedOperator/oclamp.py", line 4, in <module>
    y=flow.clamp_max(x,10)
AttributeError: module 'oneflow' has no attribute 'clamp_max'
"""