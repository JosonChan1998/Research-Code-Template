import torch
import torchvision

from utils.data.structure.bbox import BoxList
from utils.data.structure.semseg import get_semseg, SemanticSegmentation
from utils.data.structure.parse import get_parsing, Parsing
from utils.data.structure.target import Target

__all__ = ["ParsingDataset"]

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno, filter_crowd=True):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    if filter_crowd:
        # if image only has crowd annotation, it should be filtered
        if 'iscrowd' in anno[0]:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    return True

class ParsingDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self, root, ann_file, filter_no_anno, transforms=None):
        super(ParsingDataset, self).__init__(root, ann_file)

        # get image ids and filter images without detection annotations
        self.ids = sorted(self.ids)
        if filter_no_anno == True:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        # get id to img map
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        # get transforms
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(ParsingDataset, self).__getitem__(idx)

        # target: all label
        target = Target()

        # bbox label
        bboxes_anno = [obj["bbox"] for obj in anno]
        bboxes_anno = torch.as_tensor(bboxes_anno).reshape(-1, 4)
        bboxes = BoxList(bboxes_anno, img.size, mode="xywh").convert_mode("xyxy")
        target.add_field('bboxes', bboxes)

        # semseg label
        semsegs_anno = get_semseg(self.root, self.coco.loadImgs(self.ids[idx])[0]['file_name'])
        semsegs = SemanticSegmentation(semsegs_anno, img.size, mode='pic')
        target.add_field("semsegs", semsegs)

        # parsing label
        parsing_anno = [get_parsing(self.root, obj["parsing"]) for obj in anno]
        parsing = Parsing(parsing_anno, img.size)
        target.add_field("parsing", parsing)

        # transform for img and target
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        # check the bbox location in image
        target.add_field("bboxes", target.get_field("bboxes").clip_to_image())

        return img, target, idx

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data