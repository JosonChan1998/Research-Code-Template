import torch
import numpy as np
from abc import abstractmethod

__all__ = ["BaseLabel"]

class BaseLabel(object):
    """
    This class is the base class label for other label
    """

    def __init__(self, label, image_size, mode=None):
        device = label.device if isinstance(label, torch.Tensor) else torch.device("cpu")
        if isinstance(label, (list, tuple, np.ndarray, torch.Tensor)):
            label = torch.as_tensor(label, dtype=torch.float32, device=device)

        self.label = label
        self.size = image_size # (image_width, image_height)
        self.mode = mode
    
    def set_size(self, new_size):
        self.size = new_size
    
    def to(self, device):
        label = BaseLabel(self.label.to(device), self.size, self.mode)
        return label
    
    def __getitem__(self, item):
        label = BaseLabel(self.label[item], self.size, self.mode)
        return label

    def __len__(self):
        return self.label.shape[0]
    
    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_label={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s

    @abstractmethod
    def convert_mode(self, mode):
        pass
    
    @abstractmethod
    def move(self, gap):
        pass

    @abstractmethod
    def resize(self, size):
        pass

    @abstractmethod
    def corp(self, box):
        pass
    
    @abstractmethod
    def transpose(self, method):
        pass