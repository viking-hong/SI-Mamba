import torch
from torch.utils.data import DataLoader
import math
from catalyst.contrib.nn import Lookahead
from catalyst import utils

from geoseg.datasets.datasets_SI import SeaIceDataset, train_aug, val_aug
#from geoseg.models.UNetFormer1 import UNetFormer
#from geoseg.models.UNetFormer_lsk_qarepvgg import UNetFormer_lsk_s
from geoseg.models.Mamba_1 import SIMamba
from geoseg.losses import UnetFormerLoss, MambaLoss


####2/7 idx+loss+head   8 GF  1 无loss  3baseline 9/11 HY 4 无head 5只有idx 6只有head 10权重idx 12wanquan
###14 修改1 15 无loss
exp_idx = 16

# train hparam
device = 'cuda:0'
max_epoch = 100
ema_start_epoch = None
train_batch_size = 16
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
img_size = (512, 512)
resume = False
num_classes = 2
# val hparam
val_batch_size = 4 #rep
monitor = 'mIoU'

# dataloader
CLASSES = ('SeaIce', 'Background')#, 'Basckground'
ignore_index = len(CLASSES)


use_aux_loss = True
train_data_path = 'data/Sentinel-2/BH220111'
val_data_path = 'data/Sentinel-2/BH220106'

train_dataset = SeaIceDataset(data_root=train_data_path, mode='train', mosaic_ratio=0.25, transform=train_aug)
val_dataset = SeaIceDataset(data_root=val_data_path, transform=val_aug)
train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, num_workers=4,
                        pin_memory=False, shuffle=True, drop_last=True, persistent_workers=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, num_workers=4, shuffle=False,
                        pin_memory=False, drop_last=False, persistent_workers=True)

weights_path = "weights/Mamba_SI/seaice/{}".format(exp_idx)
net = SIMamba(num_classes=len(CLASSES))

# loss
loss = UnetFormerLoss(ignore_index=ignore_index)


# optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lrf = 0.01
lf = lambda x: ((1 - math.cos(x * math.pi / max_epoch)) / 2) * (lrf - 1) + 1
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

## log
log_name = 'seaice/{}'.format(exp_idx)#weights_name

## eval
eval_data_path = './data/Sentinel-2/BH220106'
eval_batch_size = 1
eval_img_size = (512, 512)
eval_weights_name = 'best.pt'
output_path = './output/exp/SI-Mamba/'