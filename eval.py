import os
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from model_v2 import ENet
from loading_data import loading_data
from utils import *
from timer import Timer
import pdb

model='encoder'

_, val_loader, __ = loading_data()

def main(): 
    torch.cuda.set_device(1)
    torch.backends.cudnn.benchmark = True

    net = []   
    
    if model=='all':
        net = ENet(only_encode=False)
        
        net.encoder.load_state_dict(encoder_weight)
    elif model =='encoder':
        net = ENet(only_encode=True)
        encoder_weight = torch.load('./ckpt/new_weight_17-12-27_10-15-08_encoder_ENet_v2_city_[320, 640]_lr_0.0005/encoder_ep_497_mIoU_0.5098.pth')
        net.encoder.load_state_dict(encoder_weight)

    net=net.cuda()
    evaluate(val_loader, net)


def evaluate(val_loader, net):
    net.eval()
    input_batches = []
    output_batches = []
    label_batches = []

    for vi, data in enumerate(val_loader, 0):

    	if vi % 10==0:
    		print vi

        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()

        outputs = net(inputs)

        input_batches.append(inputs.cpu().data)
        output_batches.append(outputs.cpu())
        label_batches.append(labels.cpu())

    input_batches = torch.cat(input_batches)
    output_batches = torch.cat(output_batches)
    label_batches = torch.cat(label_batches)

    output_batches = output_batches.cpu().data[:, :19, :, :]
    label_batches = label_batches.cpu().data.numpy()
    prediction_batches = output_batches.max(1)[1].squeeze_(1).numpy()

    mean_iu_1 = calculate_mean_iu(prediction_batches, label_batches, 19)

    print mean_iu_1



if __name__ == '__main__':
    main()








