import torch

def remove_boxes_by_center(boxlist, crop_region):
    xyxy_boxes = boxlist.convert("xyxy").bbox
    left, up, right, bottom = crop_region
    # center of boxes should inside the crop img
    centers = (xyxy_boxes[:, :2] + xyxy_boxes[:, 2:]) / 2
    keep = (
            (centers[:, 0] > left) & (centers[:, 1] > up) &
            (centers[:, 0] < right) & (centers[:, 1] < bottom)
    ).nonzero().squeeze(1)
    return boxlist[keep]


def remove_boxes_by_overlap(ori_targets, crop_targets, iou_th):
    ori_targets.size = crop_targets.size
    iou_matrix = boxlist_iou(ori_targets, crop_targets)
    iou_list = torch.diag(iou_matrix, diagonal=0)
    keep = (iou_list >= iou_th).nonzero().squeeze(1)
    return crop_targets[keep]

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou