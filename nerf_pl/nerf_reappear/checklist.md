
# CheckList about Nerf

**Format:** 编号, pytorch源文件位置=>目标文件位置，描述，是否可以顺利执行，是否对齐了输出

1. `torchsearchsorted` => `torch.searchsorted` and `flow.searchsorted`, 第二个输入的值在第一个排好序的输入张量中查找索引, [x], [x]
2. `models.nerf.Embedding` => `nerf_reappear.oneflow.Embedding`, 位置编码 (**存在pytorch未对齐问题**，链接:https://github.com/Oneflow-Inc/OneTeam/issues/1598#issue-1321799289), [x], [x]
3. `models.nerf.NeRF` =>  `nerf_reappear.oneflow.NeRF`, 将color和dir编码成rgb和sigma，其中sigma代表体密度，代表光线在某一点终止的概率 (由于全连接层具备随机性，我全部初始化为常量进行测试了), [x], [x]
4. `models.rendering.render_rays`, 通过计算应用于rays的model的输出来渲染射线(主函数), [x], []
   1. `models.rendering.render_rays.inference` => `nerf_rappear.oneflow.inference`, 表面上看上去是适应推理阶段的，但阅读代码后可以发现它支持两种形式（推理与训练），只需要控制weight_only即可 (由于Embedding发现oneflow存在一个小bug,因此我在Embedding内部.cuda了), [x], []
   2. `models.rendering.render_rays.sample_pdf` => `nerf_rappear.oneflow.sample_cdf`, 从weight的分布中采样样本 (发现**flow.clamp_min和flow.clamp_max缺失**，而torch是有的,链接：https://github.com/Oneflow-Inc/OneTeam/issues/1600), [x], [x]
5. `datasets.blender.BlenderDataset` => , Blender的Dataset，这个之后是第一步要迁移的, [], []
6. `datasets.blender.LLFFDataset` => , LLFF的Dataset，这个之后是第一步要迁移的, [], []
7. `datasets.ray_utils.get_ray_directions` => `nerf_rappear.oneflow.get_ray_directions`, 目前无法对齐（缺失create_meshgrid），但其实可以通过转换操作转过来，毕竟这边不需要梯度, [x], [x]
8. `datasets.ray_utils.get_rays` => `nerf_rappear.oneflow.get_rays`, 获取一张图像中所有像素在世界坐标中的光线原点和归一化方向, [x], [x]
9. `datasets.ray_utils.get_ndc_rays` => `nerf_rappear.oneflow.get_ndc_rays`, 将光线从世界坐标转换为 NDC (画布是一个立方体), [x], [x]
10. `datasets.depth_utils.read_pfm` and `datasets.depth_utils.save_pfm`, 无关pytorch算子，所以直接迁移即可, [x], [x]

**Note:** 大部分算子我已经迁移完成，接下来先迁移和数据集相关的算子.