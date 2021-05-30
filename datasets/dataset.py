import torch
import torchvision

from utils.data.labels import Labels
from utils.data.bounding_box import BoxList
from utils.data.semantic_segmentation import get_semseg, SemanticSegmentation

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

        # filter images without detection annotations
        self._ids = sorted(self.ids)
        if filter_no_anno == True:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self._ids = ids

        # transforms
        self._transforms = transforms   

    def __getitem__(self, idx):
        img, anno = super(ParsingDataset, self).__getitem__(idx)

        # labels
        label = Labels()

        # bbox label
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        print(boxes)
        bboxes = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        label.add_field("bboxes", bboxes)

        # semseg label
        semsegs_anno = get_semseg(self.root, self.coco.loadImgs(self.ids[idx])[0]['file_name'])
        semsegs = SemanticSegmentation(semsegs_anno, classes, img.size, mode='pic')
        label.add_field("semsegs", semsegs)

        # # parsing label
        # parsing = [get_parsing(self.root, obj["parsing"]) for obj in anno]
        # parsing = Parsing(parsing, img.size)
        # label.add_field("parsing", parsing)

        # transform for img and target
        if self._transforms is not None:
            img, label = self._transforms(img, label)

        return img, label, idx

    def __len__(self):
        return len(self._ids)

    def test(self):
        print(self._ids)