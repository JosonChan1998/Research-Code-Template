import os
import cv2
import torch
import numpy as np
from torch.nn import functional as F

from .base_label import BaseLabel

__all__ = ["SemanticSegmentation", "get_semseg"]

# transpose method
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

# class flip map
FLIP_MAP = ([14, 15], [16, 17], [18, 19])

class SemanticSegmentation(BaseLabel):
    def __init__(self, sem_seg, image_size, mode='pic'):
        super().__init__(sem_seg, image_size, mode=mode)

        # check three channel , 1xHxW
        self.label = self.label.unsqueeze(0) if len(self.label.shape) == 2 else self.label
        assert len(self.label.shape) == 3 and self.label.shape[0] == 1

    def convert_mode(self, mode):
        if mode not in ('pic'):
            raise ValueError("Only pic implemented")

    def resize(self, size):
        """
        Returns a resized copy of this semseg pic
        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)
        assert width > 0
        assert height > 0

        # Height comes first here!
        resized_semseg = F.interpolate(
            self.label[None].float(),
            size=(height, width),
            mode="nearest",
        )[0].type_as(self.label)

        resized_size = width, height
        return SemanticSegmentation(resized_semseg, resized_size, mode=self.mode)
    
    def move(self, gap):
        pass 
    
    def corp(self, box):

        assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))
        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = [round(float(b)) for b in box]

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        width, height = xmax - xmin, ymax - ymin
        cropped_semseg = self.label[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height

        return SemanticSegmentation(cropped_semseg, cropped_size, mode=self.mode)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError("Only FLIP_LEFT_RIGHT implemented")

        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_semseg = self.label.flip(dim)

        if self.mode == 'pic':
            flipped_semseg = flipped_semseg.numpy()
            for l_r in FLIP_MAP:
                left = np.where(flipped_semseg == l_r[0])
                right = np.where(flipped_semseg == l_r[1])
                flipped_semseg[left] = l_r[1]
                flipped_semseg[right] = l_r[0]
            flipped_semseg = torch.from_numpy(flipped_semseg)

        return SemanticSegmentation(flipped_semseg, self.size, mode=self.mode)

def get_semseg(root_dir, img_name):
    """
    get picture form annotations when parsing task runs
    """
    semseg_dir = root_dir.replace('img', 'seg')
    semseg_path = os.path.join(semseg_dir, img_name.replace('jpg', 'png'))
    return cv2.imread(semseg_path, 0)