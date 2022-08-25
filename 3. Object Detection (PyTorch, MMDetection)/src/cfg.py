from mmcv import Config
from mmdet.apis import set_random_seed
from dataset import XMLCustomDataset

cfg = Config.fromfile('mmdetection/configs/yolox/yolox_l_8x8_300e_coco.py')
print(f"Default Config:\n{cfg.pretty_text}")

cfg.dataset_type = 'XMLCustomDataset'
cfg.data_root = 'input/data_root/'

cfg.data.train.dataset.type = 'XMLCustomDataset'
cfg.data.train.dataset.data_root = 'input/data_root/'
cfg.data.train.dataset.ann_file = 'dataset/ImageSets/Main/train.txt'
cfg.data.train.dataset.img_prefix = 'dataset/'

cfg.data.val.type = 'XMLCustomDataset'
cfg.data.val.data_root = 'input/data_root/'
cfg.data.val.ann_file = 'dataset/ImageSets/Main/test.txt'
cfg.data.val.img_prefix = 'dataset/'

cfg.data.test.type = 'XMLCustomDataset'
cfg.data.test.data_root = 'input/data_root/'
cfg.data.test.ann_file = 'dataset/ImageSets/Main/test.txt'
cfg.data.test.img_prefix = 'dataset/'

cfg.data.train.pipeline = [
    dict(type='Mosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-320, -320)),
    dict(
        type='MixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# Batch size (samples per GPU).
cfg.data.samples_per_gpu = 2

# Modify number of classes as per the model head.
cfg.model.bbox_head.num_classes = 2
# Comment/Uncomment this to training from scratch/fine-tune according to the
# model checkpoint path.
cfg.load_from = 'checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.0008 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 5

# The output directory for training. As per the model name.
cfg.work_dir = 'outputs/yolox_l_8x8_300e_coco'
# Evaluation Metric.
cfg.evaluation.metric = 'mAP'
cfg.evaluation.save_best = 'mAP'
# Evaluation times.
cfg.evaluation.interval = 1
# Checkpoint storage interval.
cfg.checkpoint_config.interval = 10
# Set random seed for reproducible results.
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'
cfg.runner.max_epochs = 15
# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]
# We can initialize the logger for training and have a look
# at the final config used for training
print('#'*50)
print(f'Config:\n{cfg.pretty_text}')
