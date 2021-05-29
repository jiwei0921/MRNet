# -*- coding: utf-8 -*-
"""
File: demo.py
Author: Wei Ji (wji3@ualberta.ca)
Date: 2021/5/28
"""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.backends import cudnn
import os
import sys
import argparse
import logging
import numpy as np

from config import Config
from models.VGGnet.Vggnet import VGG16Net
from models.soft_attention import SoftAtt
from models.Unet.unet_model import UNet
from optimizers import init_optim
from utils.dataset_loader_cvpr import MyData
from utils.utils import Logger, mkdir_if_missing, load_pretrain_vgg16
from trainer.trainer_cvpr import Trainer
import time



parameters = dict(
        max_iteration=40000,
        spshot=30,
        nclass=2,
        b_size=8,
        sshow=655,
        phase="train",                   # train or test
        param=False,                     # Loading checkpoint
        dataset="Magrabia",              # test or val (dataset)
        snap_num=20,                     # Snapshot Number
        gpu_ids='1',                     # CUDA_VISIBLE_DEVICES
)


config = Config()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

parser=argparse.ArgumentParser()
parser.add_argument('--gpu', default=parameters["gpu_ids"], type=str, help='gpu device ids')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--arch', default='resnet34', type=str, help='backbone model name')
parser.add_argument('--b_size', default=parameters["b_size"], type=int, help='batch size for train')
parser.add_argument('--phase', default=parameters["phase"], type=str, help='train or test')
parser.add_argument('--param', default=parameters["param"], type=str, help='path to pre-trained parameters')

parser.add_argument('--train_dataroot', default='./DiscRegion', type=str, help='path to train data')
parser.add_argument('--test_dataroot', default='./DiscRegion', type=str, help='path to test or val data')

parser.add_argument('--val_root', default='./Out/val', type=str, help='directory to save run log')
parser.add_argument('--log_root', default='./Out/log', type=str, help='directory to save run log')
parser.add_argument('--snapshot_root', default='./Out/snapshot', type=str, help='path to checkpoint or snapshot')
parser.add_argument('--output_root', default='./Out/results', type=str, help='path to saliency map')
args = parser.parse_args()


torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
cuda = torch.cuda.is_available()
cudnn.benchmark = False

log_name = os.path.join(args.log_root, 'log_%s-%s.txt' % (args.phase,time.strftime("%Y-%m-%d-%H-%M-%S")))
sys.stdout = Logger(log_name)
print("==========\nArgs:{}\n==========".format(args))
torch.cuda.manual_seed_all(args.seed)
np.random.seed(5)
config.display()


"""""""""""~~~ dataset loader ~~~"""""""""
mkdir_if_missing(args.snapshot_root)
mkdir_if_missing(args.output_root)
mkdir_if_missing(args.log_root)
mkdir_if_missing(args.val_root)

train_sub = MyData(args.train_dataroot, DF=['BinRushed','MESSIDOR'],transform=True)
train_loader = DataLoader(train_sub, batch_size=args.b_size, shuffle=True, num_workers=4, pin_memory=True)
val_sub = MyData(args.test_dataroot, DF=[parameters["dataset"]])
val_loader = DataLoader(val_sub, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
test_sub = MyData(args.test_dataroot, DF=[parameters["dataset"]])
test_loader = DataLoader(test_sub, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

print('Data already.')


""""""""""" ~~~nets~~~ """""""""""
start_epoch = 0
start_iteration = 0


num_classes = parameters['nclass']
print("Initializing model: %s, n_class: %d" %(args.arch, parameters['nclass']))
logging.info(f'Network:\n'
             f'\t{num_classes} output channels (classes)')


model_rgb = UNet(resnet=args.arch, num_classes=num_classes, pretrained=False)
model_six = VGG16Net(num_classes=num_classes)
model_att = SoftAtt()


if cuda and len(args.gpu) == 1:
    model_rgb = model_rgb.cuda()
    model_six = model_six.cuda()
    model_att = model_att.cuda()
if len(args.gpu) > 1:       # multiple GPU
    model_rgb = torch.nn.DataParallel(model_rgb).cuda()
    model_six = torch.nn.DataParallel(model_six).cuda()
    model_att = torch.nn.DataParallel(model_att).cuda()
if args.param is True:      # load pretrain or checkpoint
    ckpt_path1 = os.path.join(args.snapshot_root, 'snapshot_iter_'+ str(parameters['snap_num'])+'.pth')
    ckpt_path2 = os.path.join(args.snapshot_root, 'six_snapshot_iter_' + str(parameters['snap_num']) + '.pth')
    ckpt_path3 = os.path.join(args.snapshot_root, 'att_snapshot_iter_' + str(parameters['snap_num']) + '.pth')
    model_rgb.load_state_dict(torch.load(ckpt_path1, map_location='cpu'))
    model_six.load_state_dict(torch.load(ckpt_path2, map_location='cpu'))
    model_att.load_state_dict(torch.load(ckpt_path3, map_location='cpu'))
else:
    if args.arch =='vgg16':
        load_pretrain_vgg16(model_rgb,pretrain=False)



""""""""""" ~~~train or test~~~ """""""""

#Trainer: class, defined in trainer.py
optimizer_rgb = init_optim(config.OPTIMIZERS, model_rgb.parameters(), config.LR, config.WEIGHT_DECAY)
optimizer_six = init_optim(config.OPTIMIZERS, model_six.parameters(), config.LR, config.WEIGHT_DECAY)
optimizer_att = init_optim(config.OPTIMIZERS, model_att.parameters(), config.LR, config.WEIGHT_DECAY)
training = Trainer(
        cuda=cuda,
        model_rgb=model_rgb,
        model_six=model_six,
        model_att=model_att,
        optimizer_rgb=optimizer_rgb,
        optimizer_six=optimizer_six,
        optimizer_att=optimizer_att,
        train_sub=train_sub,
        val_sub=val_sub,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_sub=test_sub,
        max_iter=parameters['max_iteration'],
        snapshot=parameters['spshot'],
        outpath=args.snapshot_root,
        sshow=parameters['sshow'],
        step_size=config.STEP_SIZE,
        gamma=config.GAMMA,
        log_name=log_name,
        val_out_path=[args.val_root, args.test_dataroot],
    )



if args.phase == 'train':
    training.epoch = start_epoch
    training.iteration = start_iteration
    training.train()

elif args.phase == 'test':
    training.val_epoch(epoch=1,val_flag=False)





