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
    0.75, 1.333,
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

cfg.anchor_shapes = [ \
	1.155, 3.464,
	1.414, 2.828,
	2.0, 2.0,
	2.828, 1.414,
	3.464, 1.155,
	1.732, 5.196,
	2.121, 4.243,
	3.0, 3.0,
	4.243, 2.121,
	5.196, 1.732,
	3.464, 10.392,
	4.243, 8.485,
	6.0, 6.0,
	8.485, 4.243,
	10.392, 3.464,
	5.774, 17.321,
	7.071, 14.142,
	10.0, 10.0,
	14.142, 7.071,
	17.321, 5.774,
	11.547, 34.641,
	14.142, 28.284,
	20.0, 20.0,
	28.284, 14.142,
	34.641, 11.547,
	19.799, 39.598,
	28.0, 28.0,
	39.598, 19.799,
    ] # 544

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
cfg.train.rand_pad = RandPadder(rand_pad_prob=0.5, max_pad_scale=4.0)
# random color jitter
cfg.train.color_jitter = ColorJitter(random_hue_prob=0.5, random_saturation_prob=0.5,
    random_illumination_prob=0.5, random_contrast_prob=0.5)
cfg.train.inter_method = 10  # random interpolation
cfg.train.rand_mirror_prob = 0.5
cfg.train.shuffle = True
cfg.train.seed = 233
cfg.train.preprocess_threads = 48

cfg.train.focal_loss_alpha = 0.25
cfg.train.focal_loss_alpha_rpn = 0.5
cfg.train.focal_loss_gamma = 2.0
cfg.train.smoothl1_weight = 1.0

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
