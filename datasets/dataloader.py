import copy
import bisect
import torch.utils.data
import torch.distributed as dist

from cfgs.CIHP_cfg import cfg
from utils.data import samplers
from utils.data.collate_batch import BatchCollator

__all__ = ["get_train_data_loader", "get_test_data_loader"]

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        if cfg.DATALOADER_SAMPLER_TRAIN == "RepeatFactorTrainingSampler":
            return samplers.RepeatFactorTrainingSampler(dataset, cfg.DATALOADER.RFTSAMPLER, shuffle=shuffle)
        else:
            return samplers.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler

def get_train_data_loader(datasets, is_distributed=False, start_iter=0):

    if dist.is_available() or dist.is_initialized():
        num_gpus = 1
    else:
        num_gpus = dist.get_world_size()

    ims_per_gpu = int(cfg.TRAIN_BATCH_SIZE / num_gpus)
    shuffle = True
    num_iters = cfg.SOLVER_MAX_ITER

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER_ASPECT_RATIO_GROUPING else []

    sampler = make_data_sampler(datasets, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        datasets, sampler, aspect_grouping, ims_per_gpu, num_iters, start_iter
    )
    collator = BatchCollator(cfg.TRAIN_SIZE_DIVISIBILITY)
    num_workers = cfg.TRAIN_LOADER_THREADS
    data_loader = torch.utils.data.DataLoader(
        datasets,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )

    return data_loader

def get_test_data_loader(datasets, start_ind, end_ind, is_distributed=True):
    ims_per_gpu = cfg.TEST.IMS_PER_GPU
    if start_ind == -1 or end_ind == -1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(datasets) if is_distributed else None
    else:
        test_sampler = samplers.RangeSampler(start_ind, end_ind)
    num_workers = cfg.TEST.LOADER_THREADS
    collator = BatchCollator(cfg.TEST.SIZE_DIVISIBILITY)
    data_loader = torch.utils.data.DataLoader(
        datasets,
        batch_size=ims_per_gpu,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collator,
    )

    return data_loader
