from torch.utils import data
import _init_paths

from datasets.dataset import ParsingDataset

root = "/home/sunyanxiao/josonchan/CIHP/train_img"
ann_file = "/home/sunyanxiao/josonchan/CIHP/CIHP_train.json"

dataset = ParsingDataset(root, ann_file, filter_no_anno=True)

print(dataset.__getitem__(1))


