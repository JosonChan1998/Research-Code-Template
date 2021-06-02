import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base_label import BaseLabel

__all__ = ["Parsing"]

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

FLIP_MAP = ([14, 15], [16, 17], [18, 19])

class Parsing(BaseLabel):
    def __init__(self, parsing, image_size, mode=None):
        super().__init__(parsing, image_size, mode=mode)

        # if only a single instance mask is passed, change to 1xHxW
        if len(self.label.shape) == 2:
            self.label = self.label.unsqueeze(0)

        # check the label shape 1xHxW
        assert len(self.label.shape) == 3
        assert self.label.shape[1] == image_size[1], "%s != %s" % (self.label.shape[1], image_size[1])
        assert self.label.shape[2] == image_size[0], "%s != %s" % (self.label.shape[2], image_size[0])

    def convert_mode(self, mode):
        pass

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)
        assert width > 0
        assert height > 0

        # Height comes first here!, change 1xhxw to 1x1xhxw
        resized_parsing = F.interpolate(
            self.label.unsqueeze(0).float(),
            size=(height, width),
            mode="nearest",
        )[0].type_as(self.label)

        resized_size = width, height
        return Parsing(resized_parsing, resized_size)

    def move(self, gap):
        c, h, w = self.label.shape
        old_up, old_left, old_bottom, old_right = max(gap[1], 0), max(gap[0], 0), h, w

        new_up, new_left = max(0 - gap[1], 0), max(0 - gap[0], 0)
        new_bottom, new_right = h + new_up - old_up, w + new_left - old_left
        new_shape = (c, h + new_up, w + new_left)

        moved_parsing = torch.zeros(new_shape, dtype=torch.uint8)
        moved_parsing[:, new_up:new_bottom, new_left:new_right] = \
            self.label[:, old_up:old_bottom, old_left:old_right]

        moved_size = new_shape[2], new_shape[1]
        return Parsing(moved_parsing, moved_size)

    def crop(self, box):
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
        cropped_parsing = self.label[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height
        return Parsing(cropped_parsing, cropped_size)
    
    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError("Only FLIP_LEFT_RIGHT implemented")

        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_parsing = self.label.flip(dim)

        flipped_parsing = flipped_parsing.numpy()
        for l_r in FLIP_MAP:
            left = np.where(flipped_parsing == l_r[0])
            right = np.where(flipped_parsing == l_r[1])
            flipped_parsing[left] = l_r[1]
            flipped_parsing[right] = l_r[0]
        flipped_parsing = torch.from_numpy(flipped_parsing)
    
        return Parsing(flipped_parsing, self.size)
    

def get_parsing(root_dir, parsing_name):
    parsing_dir = root_dir.replace('img', 'parsing')
    parsing_path = os.path.join(parsing_dir, parsing_name)
    return cv2.imread(parsing_path, 0)