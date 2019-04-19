import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets import my_voc_dataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
print("current path:{}".format(os.getcwd()))
'''
python train_ssd.py  --net mb2-ssd-lite   --batch_size 24 --num_epochs 200 --scheduler cosine --lr 0.01 --t_max 200
'''
balance_data = False
base_net = None
base_net_lr = None #initial learning rate for base net.
batch_size = 24
checkpoint_folder = './models/'
debug_steps = 100
extra_layers_lr = None #initial learning rate for the layers not in base net and prediction heads.
freeze_base_net = False
freeze_net = False
gamma = 0.1 #Gamma update for SGD
lr = 0.01
mb2_width_mult = 1.0
milestones = '80,100'
momentum = 0.9
net_name = 'mb2-ssd-lite'#The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite or vgg16-ssd.
num_epochs = 200
num_workers = 0
pretrained_ssd = None
resume = None
scheduler = 'cosine'
t_max = 200.0
use_cuda = True
validation_epochs = 5
weight_decay = 0.0005#Weight decay for SGD
##
train_filename = 'E:/git/VOC0712_dataset/train.txt' #训练文件
val_filename = 'E:/git/VOC0712_dataset/val.txt'     #测试文件
num_classes = 21                                    #样本个数
dataset_type = 'voc'


logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
if use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use CUDA.")
else:
    logging.info("Use CPU.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

def select_net(net_name):
    if net_name == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif net_name == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif net_name == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif net_name == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif net_name == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=mb2_width_mult)
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        # parser.print_help(sys.stderr)
        sys.exit(1)
    return create_net,config

if __name__ == '__main__':
    timer = Timer()
    create_net,config=select_net(net_name)

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    logging.info("Prepare training datasets.")
    # for dataset_path in args.datasets:
    train_dataset = my_voc_dataset.Dataset(train_filename, transform=train_transform, target_transform=target_transform)
    # train_dataset = ConcatDataset(train_dataset)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size, num_workers=num_workers,shuffle=True,drop_last=True)
    logging.info("Prepare Validation datasets.")

    val_dataset = my_voc_dataset.Dataset(val_filename, transform=test_transform,target_transform=target_transform)
    logging.info("validation dataset size: {}".format(len(val_dataset)))
    val_loader = DataLoader(val_dataset, batch_size,num_workers=num_workers,shuffle=False,drop_last=True)
    logging.info("Build network.")

    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = base_net_lr if base_net_lr is not None else lr
    extra_layers_lr = extra_layers_lr if extra_layers_lr is not None else lr
    if freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if resume:
        logging.info(f"Resume from the model {resume}")
        net.load(resume)
    elif base_net:
        logging.info(f"Init from base net {base_net}")
        net.init_from_base_net(base_net)
    elif pretrained_ssd:
        logging.info(f"Init from pretrained ssd {pretrained_ssd}")
        net.init_from_pretrained_ssd(pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    logging.info(f"Learning rate: {lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {scheduler}.")
        # parser.print_help(sys.stderr)
        sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, num_epochs):
        scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=debug_steps, epoch=epoch)

        if epoch % validation_epochs == 0 or epoch == num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(checkpoint_folder, f"{net_name}-Epoch-{epoch}-Loss-{val_loss}.pth")
            print('model_path:{}'.format(model_path))
            net.save(model_path)
            logging.info(f"Saved model {model_path}")