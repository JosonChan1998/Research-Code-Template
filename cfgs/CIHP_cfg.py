
__all__ = ["cfg"]

class Config(object):
    # ---------------------------------------------------------------------------- #
    # Dataset options
    # ---------------------------------------------------------------------------- #
    
    # dataset root
    TRAIN_ROOT = "/home/sunyanxiao/josonchan/CIHP/train_img"
    TRAIN_ANN_ILE = "/home/sunyanxiao/josonchan/CIHP/CIHP_train.json"
    TEST_ROOT = "/home/sunyanxiao/josonchan/CIHP/val_img"
    TEST_ANN_FILE = "/home/sunyanxiao/josonchan/CIHP/CIHP_val.json"

    # Trainset options
    DATALOADER_SAMPLER_TRAIN = "DistributedSampler"
    TRAIN_BATCH_SIZE = 16
    DATALOADER_ASPECT_RATIO_GROUPING = True
    TRAIN_LOADER_THREADS = 4
    TRAIN_SIZE_DIVISIBILITY = 32

    # Testset options
    TEST_IMS_PER_GPU = 1
    TEST_LOADER_THREADS = 4
    TEST_SIZE_DIVISIBILITY = 32

    # ---------------------------------------------------------------------------- #
    # Transforms options
    # ---------------------------------------------------------------------------- #

    # Normalize
    TO_BGR255 = True
    PIXEL_MEANS = [102.9801, 115.9465, 122.7717]
    PIXEL_STDS = [1.0, 1.0, 1.0]

    # Color ColorJitter 
    TRAIN_BRIGHTNESS = 0.0
    TRAIN_CONTRAST = 0.0
    TRAIN_SATURATION = 0.0
    TRAIN_HUE = 0.0
    TRAIN_LEFT_RIGHT = ()

    # Resize 
    TRAIN_SCALE = (512, 640, 704, 768, 800, 864)
    TRAIN_MAX_SIZE = 1400
    TRAIN_RESIZE_SCALE_RATIOS = (0.8, 1.2)

    TEST_SCALE = 800
    TEST_MAX_SIZE = 1333    
    TEST_FORCE_TEST_SCALE = [-1, -1]

    # Random Crop options
    TRAIN_PREPROCESS_TYPE = 'none'
    TRAIN_RANDOM_CROP_CROP_SCALES = ([640, 640], )
    TRAIN_RANDOM_CROP_IOU_THS = (0.9, 0.7, 0.5, 0.3, 0.1)
    TRAIN_RANDOM_CROP_PAD_PIXEL = ()

    # ---------------------------------------------------------------------------- #
    # Model options
    # ---------------------------------------------------------------------------- #

    # ResNet
    RESNET_USE_ALIGN = False
    RESNET_BOTTLENECK = True
    RESNET_STRIDE_3X3 = False
    RESNET_AVG_DOWN = False
    RESNET_LAYERS = (3, 4, 6, 3)
    RESNET_WIDTH = 64
    RESNET_STAGE_WITH_CONTEXT = ('none', 'none', 'none', 'none')
    RESNET_CTX_RATIO = 0.0625
    RESNET_STAGE_WITH_CONV = ('normal', 'normal', 'normal', 'normal')
    RESNET_C5_DILATION = 1 # may be litte different from resnet
    RESNET_USE_3x3x3HEAD = False
    RESNET_FREEZE_AT = 2


    # ---------------------------------------------------------------------------- #
    # Solver options
    # ---------------------------------------------------------------------------- #

    SOLVER_MAX_ITER = 135000

cfg = Config()