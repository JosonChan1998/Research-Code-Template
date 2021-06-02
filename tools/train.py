import init_paths

from datasets.dataset import ParsingDataset
from datasets.transforms import build_transforms
from datasets.dataloader import get_train_data_loader
from models.build_model import RPRCNN

from cfgs.CIHP_cfg import cfg

data_transforms = build_transforms(is_train=True)
dataset = ParsingDataset(cfg.TRAIN_ROOT, cfg.TRAIN_ANN_ILE, filter_no_anno=True, transforms=data_transforms)
train_data_loader = get_train_data_loader(dataset, is_distributed=False)

model = RPRCNN()

def train(data_loader):

    for iteration, (images, targets, _) in enumerate(data_loader):
        images = images.to('cuda')
        targets = [target.to('cuda') for target in targets]


