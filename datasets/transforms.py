import random
import torchvision
from PIL import Image
from torchvision.transforms import functional as F

from utils.data.boxlist_ops import remove_boxes_by_center, remove_boxes_by_overlap
from cfgs.CIHP_cfg import cfg

__all__ = ["build_transforms"]

# all class add return target option
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size, preprocess_type, scale_ratios,
                 force_test_scale=[-1, -1]):
        assert preprocess_type in ["none", "random_resize"]
        assert not (preprocess_type == "none" and min_size == -1)

        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.preprocess_type = preprocess_type
        self.scale_ratios = scale_ratios
        self.force_test_scale = force_test_scale

    def reset_size(self, image_size, based_scale_size):
        if self.min_size != -1:
            h, w = based_scale_size
        else:
            h, w = image_size

        if len(self.scale_ratios) == 1:
            scale_ratio = 1
        else:
            scale_ratio = random.uniform(self.scale_ratios[0], self.scale_ratios[1])

        reset_scale_h = int(h * scale_ratio)
        reset_scale_w = int(w * scale_ratio)
        return (reset_scale_h, reset_scale_w)

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        min_size = random.choice(self.min_size)
        max_size = self.max_size

        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * min_size > max_size:
                min_size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == min_size) or (h <= w and h == min_size):
            return (h, w)

        if w < h:
            ow = min_size
            oh = int(min_size * h / w)
        else:
            oh = min_size
            ow = int(min_size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        if -1 not in self.force_test_scale:
            size = tuple(self.force_test_scale)
        else:
            size = self.get_size(image.size)
            if self.preprocess_type == "random_resize":
                size = self.reset_size(image.size, size)
        image = F.resize(image, size)
        target = target.resize(size)
        return image, target


class RandomCrop(object):
    def __init__(self, preprocess_type, crop_sizes, pad_pixel=(0, 0, 0), iou_ths=(0.7, )):
        assert preprocess_type in ["none", "random_crop"]
        self.preprocess_type = preprocess_type
        self.crop_sizes = crop_sizes
        self.pad_pixel = tuple(map(int, map(round, pad_pixel)))
        self.iou_ths = iou_ths

    def get_crop_coordinate(self, image_size):
        w, h = image_size
        crop_h, crop_w = random.choice(self.crop_sizes)

        left_change, up_change = w - crop_w, h - crop_h
        left = random.randint(min(0, left_change), max(0, left_change))
        up = random.randint(min(0, up_change), max(0, up_change))

        crop_region = (left, up, min(w, left + crop_w), min(h, up + crop_h))
        crop_shape = (crop_w, crop_h)
        return crop_region, crop_shape

    def image_crop_with_padding(self, img, crop_region, crop_shape):
        set_left, set_up, right, bottom = crop_region
        crop_left, corp_up = max(set_left, 0), max(set_up, 0)
        crop_region = (crop_left, corp_up, right, bottom)

        img = img.crop(crop_region)
        if img.size != crop_shape:
            pad_img = Image.new('RGB', crop_shape, self.pad_pixel)
            paste_region = (max(0-set_left, 0),
                            max(0-set_up, 0),
                            max(0-set_left, 0)+img.size[0],
                            max(0-set_up, 0)+img.size[1])
            pad_img.paste(img, paste_region)
            return pad_img

        return img

    def targets_crop(self, targets, crop_region, crop_shape):
        # label move
        set_left, set_up, right, bottom = crop_region
        targets = targets.move((set_left, set_up))

        # determine the bbox or parsing bbox is in the crop_image
        reset_region = (0, 0, min(right-min(set_left, 0), crop_shape[0])-1,
                        min(bottom-min(set_up, 0), crop_shape[1])-1)
        targets.add_field("bboxes", 
                            remove_boxes_by_center(
                            targets.get_field("bboxes"), 
                            reset_region))

        crop_targets = targets.crop(reset_region)

        # determine the bbox iou greater than iou_th
        iou_th = random.choice(self.iou_ths)
        targets.add_field("bboxes",
                            remove_boxes_by_overlap(
                            targets.get_field("bboxes"), 
                            crop_targets, iou_th))
                            
        targets = targets.set_size(crop_shape)
        return targets

    def __call__(self, image, targets):
        if self.preprocess_type == "none":
            return image, targets
        crop_region, crop_shape = self.get_crop_coordinate(image.size)
        out_image = self.image_crop_with_padding(image, crop_region, crop_shape)
        out_targets = self.targets_crop(targets, crop_region, crop_shape)
        # if crop_region don't have instance，random crop again.
        if len(out_targets) == 0:
            return self.__call__(image, targets)
        return out_image, out_targets


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5, left_right=()):
        self.prob = prob
        self.left_right = left_right

    def __call__(self, image, targets):
        if random.random() < self.prob:
            image = F.hflip(image)
            targets = targets.transpose(method=0)
        return image, targets


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

    

def build_transforms(is_train=True):
    if is_train:
        min_size = cfg.TRAIN_SCALE
        max_size = cfg.TRAIN_MAX_SIZE
        flip_prob = 0.5

        brightness = cfg.TRAIN_BRIGHTNESS
        contrast = cfg.TRAIN_CONTRAST
        saturation = cfg.TRAIN_SATURATION
        hue = cfg.TRAIN_HUE
        left_right = cfg.TRAIN_LEFT_RIGHT

        # for force resize
        force_test_scale = [-1, -1]
        scale_ratios = cfg.TRAIN_RESIZE_SCALE_RATIOS

        # for random crop
        preprocess_type = cfg.TRAIN_PREPROCESS_TYPE
        crop_sizes = cfg.TRAIN_RANDOM_CROP_CROP_SCALES
        crop_iou_ths = cfg.TRAIN_RANDOM_CROP_IOU_THS
        pad_pixel = cfg.TRAIN_RANDOM_CROP_PAD_PIXEL
        pad_pixel = (cfg.PIXEL_MEANS if len(pad_pixel) < 3 else pad_pixel)
    else:
        min_size = cfg.TEST_SCALE
        max_size = cfg.TEST_MAX_SIZE
        flip_prob = 0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0
        left_right = ()

        # for force resize
        force_test_scale = cfg.TEST_FORCE_TEST_SCALE
        scale_ratios = ()

        # for random crop
        preprocess_type = "none"
        crop_sizes = ()
        pad_pixel = ()
        crop_iou_ths = ()

    to_bgr255 = cfg.TO_BGR255
    normalize_transform = Normalize(
        mean=cfg.PIXEL_MEANS, std=cfg.PIXEL_STDS, to_bgr255=to_bgr255
    )

    color_jitter = ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = Compose(
        [
            color_jitter,
            Resize(min_size, max_size, preprocess_type, scale_ratios, force_test_scale),
            RandomCrop(preprocess_type, crop_sizes, pad_pixel, crop_iou_ths),
            RandomHorizontalFlip(flip_prob, left_right),
            ToTensor(),
            normalize_transform,
        ]
    )
    return transform