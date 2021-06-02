from numpy.lib.type_check import imag
import torch
import init_paths

from cfgs.CIHP_cfg import cfg

from datasets.new_dataset import ParsingDataset
from datasets.transforms import build_transforms
from datasets.dataloader import get_train_data_loader


data_transforms = build_transforms(is_train=True)
dataset = ParsingDataset(cfg.ROOT, cfg.ANN_ILE, filter_no_anno=True, transforms=data_transforms)
train_data_loader = get_train_data_loader(dataset, is_distributed=False)

for iteration, (images, targets, _) in enumerate(train_data_loader):
    pass

