import os
import sys
import time
parent_path = os.path.dirname(os.getcwd())
if 'tools' in os.getcwd():
    os.chdir(parent_path)
sys.path.insert(0, os.getcwd())

import json
import torch
import torch.nn as nn
from torchvision import transforms
from dataloader.KTHDataset.KTHDataset import KTHDataset

import argparse

import logging
from utils.log_helper import init_log, add_file_handler, print_speed
from utils.config_helper import Configs
from utils.average_meter_helper import AverageMeter
from model.loss.SSIM_Loss import SSIM
from model.loss.L1_L2_Loss import L1_L2_Loss

import inspect
from utils.memory.gpu_mem_track import MemTracker
frame = inspect.currentframe()
gpu_tracker = MemTracker(frame)

# 生成命令行的参数
parser = argparse.ArgumentParser(description='Train moving mnist video prediction algorithm')
parser.add_argument('-c', '--cfg', default=os.path.join(os.getcwd(), "tools", "train_config_PredRNN.json"), type=str, required=False, help='training config file path')

args = parser.parse_args()

# 初始化一些变量
cfg = Configs(args.cfg)
# board的路径
board_path = cfg.meta["board_path"]
experiment_path = cfg.meta["experiment_path"]
experiment_name = cfg.meta["experiment_name"]
arch = cfg.meta["arch"]
# 训练时候的一些参数
batch_size = cfg.train['batch_size']
epoches = cfg.train['epoches']
lr = cfg.train['lr']
# 初始化未来帧的数量
num_frame = cfg.model['input_num']
# print freq
print_freq = cfg.train['print_freq']

# 初始化logger
global_logger = init_log('global', level=logging.INFO)
add_file_handler("global", os.path.join(os.getcwd(), 'logs', '{}.log'.format(experiment_name)), level=logging.DEBUG)

# 打印cfg信息
cfg.log_dict()

# 初始化avrager
avg = AverageMeter()

# cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# 准备数据集
train_set = KTHDataset(root='./data/KTHDataset', train=True, download=True,
                        transform=transforms.Compose([transforms.Resize(cfg.model["input_size"]), transforms.ToTensor(),]),
                        target_transform=transforms.Compose([transforms.Resize(cfg.model["input_size"]), transforms.ToTensor(),]))
test_set = KTHDataset(root='./data/KTHDataset', train=False, download=True,
                        transform=transforms.Compose([transforms.Resize(cfg.model["input_size"]), transforms.ToTensor(),]),
                        target_transform=transforms.Compose([transforms.Resize(cfg.model["input_size"]), transforms.ToTensor(),]))

# 建立dataloader
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=True)

# 建立test的iter
test_iter = iter(test_loader)

# 确定数据集长度
train_lenth = len(train_loader)
test_lenth = len(test_loader)

global_logger.debug('==>>> total trainning batch number: {}'.format(train_lenth))
global_logger.debug('==>>> total testing batch number: {}'.format(test_lenth))

# 加载模型
sys.path.append(os.path.join(".", experiment_path, experiment_name))
if arch == "Custom":
    from custom import Custom
    model = Custom(cfg=cfg.model)
    model = model.to(device)
    model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).to(device)
else:
    raise NotImplementedError

# 建立tensorboard的实例
from tensorboardX import SummaryWriter 
writer = SummaryWriter(os.path.join(".", board_path, experiment_name))

# 建立优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10,20,30,40], gamma=0.5)

# 建立loss
loss_L1 = nn.MSELoss().to(device)
loss_L2 = nn.L1Loss().to(device)
# crossEntropy
loss_BCE = nn.BCELoss().to(device)
# SSIM
loss_SSIM = SSIM(window_size=11, size_average=True)

loss_L1_L2 = L1_L2_Loss().to(device)

# 训练的部分
for epoch in range(epoches):
    for step, [seq, seq_target] in enumerate(train_loader):
        step_time = time.time()
        # 打印测试信息
        if epoch is 0 and step is 0:
            global_logger.debug('Input:  {}'.format(seq.shape))
            global_logger.debug('--- Sample')
            global_logger.debug('Target: {}'.format(seq_target.shape))

        # 放到cuda中
        seq, seq_target = seq.to(device), seq_target.to(device)

        # 优化器归零
        optimizer.zero_grad()

        # 送入模型进行推断
        layer_output = model(seq, future=num_frame)

        # loss计算
        train_loss = loss_BCE(layer_output[:, -num_frame:, :, :, :], seq_target[:, -num_frame:, :, :, :])
        with torch.no_grad():
            train_metric = loss_SSIM(layer_output[:, -num_frame:, :, :, :], seq_target[:, -num_frame:, :, :, :])
        train_loss.backward()

        # 优化器更新
        optimizer.step()

        # validate

        with torch.no_grad():
            # load random test set
            try:
                seq_test, gt_seq_test = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                seq_test, gt_seq_test = next(test_iter)
            
            seq_test, gt_seq_test = seq_test.to(device), gt_seq_test.to(device)

            # 送入模型进行推断
            test_output = model(seq_test, future=num_frame)

            # loss计算
            test_loss = loss_BCE(test_output[:, -num_frame:, :, :, :], gt_seq_test[:, -num_frame:, :, :, :])
            test_metric = loss_SSIM(test_output[:, -num_frame:, :, :, :], gt_seq_test[:, -num_frame:, :, :, :])

        step_time = time.time() - step_time

        # 将有用的信息存进tensorboard中
        if (step+1) % 200 == 0:
            writer.add_video('train_seq/feed_seq', seq, epoch*train_lenth + step + 1)
            writer.add_video('train_seq/gt_seq', seq_target, epoch*train_lenth + step + 1)
            writer.add_video('train_seq/pred_seq', layer_output, epoch*train_lenth + step + 1)
            writer.add_video('test_seq/feed_seq', seq_test, epoch*train_lenth + step + 1)
            writer.add_video('test_seq/gt_seq', gt_seq_test, epoch*train_lenth + step + 1)
            writer.add_video('test_seq/pred_seq', test_output, epoch*train_lenth + step + 1)
        writer.add_scalars('loss/merge', {"train_loss": train_loss,"test_loss":test_loss, "train_metric":train_metric, "test_metric":test_metric}, epoch*train_lenth + step + 1)

        # 更新avrager
        avg.update(step_time=step_time, train_loss=train_loss, test_loss=test_loss, train_metric=train_metric) # 算平均值

        # 打印结果
        if (step+1) % print_freq == 0:
                global_logger.info('Epoch: [{0}][{1}/{2}] {step_time:s}\t{train_loss:s}\t{test_loss:s}\t{train_metric:s}'.format(
                            epoch+1, (step + 1) % train_lenth, train_lenth, step_time=avg.step_time, train_loss=avg.train_loss, test_loss=avg.test_loss, train_metric=avg.train_metric))
                print_speed(epoch*train_lenth + step + 1, avg.step_time.avg, epoches * train_lenth)

    # scheduler更新
    scheduler.step()


        
