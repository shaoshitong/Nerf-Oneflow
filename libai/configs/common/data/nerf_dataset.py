from omegaconf import OmegaConf
from flowvision import transforms
from flowvision.transforms import InterpolationMode
from flowvision.transforms.functional import str_to_interp_mode
from flowvision.data.auto_augment import rand_augment_transform
from libai.config import LazyCall
from libai.data.datasets import get_nerf_dataset
from libai.data.build import build_image_train_loader, build_image_test_loader

dataset_type = "Blender"

dataset = LazyCall(get_nerf_dataset)(dataset_type=dataset_type)

dataloader = OmegaConf.create()

dataloader.train = LazyCall(build_image_train_loader)(
    dataset=[
        LazyCall(dataset)(
            split="train",
            img_wh=(400, 400) if dataset.dataset_type == "Blender" else (504, 378),
            root_dir="/path/to/blender" if dataset.dataset_type == "Blender" else "/path/to/llff",
            spheric_poses=None if dataset.dataset_type == "Blender" else False,
            val_num=None if dataset.dataset_type == "Blender" else 1,  # Number of your GPUs
        )
    ],
    num_workers=4,
)


dataloader.test = [
    LazyCall(build_image_test_loader)(
        dataset=LazyCall(dataset)(
            split="val",
            img_wh=(400, 400) if dataset.dataset_type == "Blender" else (504, 378),
            root_dir="/path/to/blender" if dataset.dataset_type == "Blender" else "/path/to/llff",
            spheric_poses=None if dataset.dataset_type == "Blender" else False,
            val_num=None if dataset.dataset_type == "Blender" else 1,  # Number of your GPUs
        ),
        num_workers=4,
    )
]
