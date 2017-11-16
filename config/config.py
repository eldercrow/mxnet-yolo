import os
from utils import DotDict, namedtuple_with_defaults, zip_namedtuple, config_as_dict

RandCropper = namedtuple_with_defaults('RandCropper',
    'min_crop_scales, max_crop_scales, \
    min_crop_aspect_ratios, max_crop_aspect_ratios, \
    min_crop_overlaps, max_crop_overlaps, \
    min_crop_sample_coverages, max_crop_sample_coverages, \
    min_crop_object_coverages, max_crop_object_coverages, \
    max_crop_trials',
    [0.0, 1.0,
    0.5, 2.0,
    0.0, 1.0,
    0.0, 1.0,
    0.0, 1.0,
    25])

RandPadder = namedtuple_with_defaults('RandPadder',
    'rand_pad_prob, max_pad_scale, fill_value',
    [0.0, 1.0, 127])

ColorJitter = namedtuple_with_defaults('ColorJitter',
    'random_hue_prob, max_random_hue, \
    random_saturation_prob, max_random_saturation, \
    random_illumination_prob, max_random_illumination, \
    random_contrast_prob, max_random_contrast',
    [0.0, 18,
    0.0, 32,
    0.0, 32,
    0.0, 0.5])


cfg = DotDict()
cfg.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# cfg.anchor_shapes = [ \
#         0.954, 1.348, \
#         1.882, 4.035, \
#         6.331, 3.807, \
#         3.985, 8.042, \
#         9.701, 9.64, \
#         ]
cfg.anchor_shapes = [ \
        0.41, 0.677,
        0.591, 1.665,
        1.17, 0.89,
        0.858, 3.29,
        1.487, 1.974,
        2.696, 1.344,
        1.811, 3.469,
        3.237, 2.853,
        1.482, 6.317,
        2.691, 5.028,
        7.234, 2.404,
        4.998, 4.427,
        2.672, 9.188,
        3.905, 6.838,
        6.725, 6.928,
        10.196, 4.643,
        4.786, 10.235,
        7.644, 10.883,
        10.901, 7.802,
        11.525, 11.587,
        ]

# training configs
cfg.train = DotDict()
# random cropping samplers
cfg.train.rand_crop_samplers = [
    RandCropper(min_crop_scales=0.5, min_crop_overlaps=0.1),
    RandCropper(min_crop_scales=0.5, min_crop_overlaps=0.3),
    RandCropper(min_crop_scales=0.5, min_crop_overlaps=0.5),
    RandCropper(min_crop_scales=0.5, min_crop_overlaps=0.7),
    RandCropper(min_crop_scales=0.5, min_crop_overlaps=0.9),]
cfg.train.crop_emit_mode = 'center'
# cfg.train.emit_overlap_thresh = 0.4
# random padding
cfg.train.rand_pad = RandPadder(rand_pad_prob=0.5, max_pad_scale=3.0)
# random color jitter
cfg.train.color_jitter = ColorJitter(random_hue_prob=0.5, random_saturation_prob=0.5,
    random_illumination_prob=0.5, random_contrast_prob=0.5)
cfg.train.inter_method = 10  # random interpolation
cfg.train.rand_mirror_prob = 0.5
cfg.train.shuffle = True
cfg.train.seed = 233
cfg.train.preprocess_threads = 48

cfg.train.focal_loss_alpha = 1.0/4.0
cfg.train.focal_loss_alpha_rpn = 0.9
cfg.train.focal_loss_gamma = 2.0
cfg.train.smoothl1_weight = 1.0
cfg.train.use_smooth_ce = False
cfg.train.smooth_ce_th = 1e-02
cfg.train.smooth_ce_lambda = 1.0

cfg.train = config_as_dict(cfg.train)  # convert to normal dict

# validation
cfg.valid = DotDict()
cfg.valid.rand_crop_samplers = []
cfg.valid.rand_pad = RandPadder()
cfg.valid.color_jitter = ColorJitter()
cfg.valid.rand_mirror_prob = 0
cfg.valid.shuffle = False
cfg.valid.seed = 0
cfg.valid.preprocess_threads = 32
cfg.valid = config_as_dict(cfg.valid)  # convert to normal dict
