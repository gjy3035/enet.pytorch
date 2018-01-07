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

from model import ENet
from config import cfg
from loading_data import loading_data
from utils import *
from timer import Timer
import pdb

exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()

train_record = {'best_val_mean_iu': -1.0, 'corr_loss': 0, 'corr_epoch': -1, 'best_model_name': ''}

train_loader, val_loader, restore_transform = loading_data()

def main():

    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = []   
    
    if cfg.TRAIN.STAGE=='all':
        net = ENet(only_encode=False)
        if cfg.TRAIN.PRETRAINED_ENCODER != '':
            encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
            del encoder_weight['classifier.bias']
            del encoder_weight['classifier.weight']
            # pdb.set_trace()
            net.encoder.load_state_dict(encoder_weight)
    elif cfg.TRAIN.STAGE =='encoder':
        net = ENet(only_encode=True)

    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.cuda()

    net.train()
    criterion = CrossEntropyLoss2d(cfg.TRAIN.LABEL_WEIGHT).cuda()
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # optimizer = optim.Adam([
    #                {'params': net.encoder.parameters(), 'lr':cfg.TRAIN.LR, 'weight_decay':cfg.TRAIN.WEIGHT_DECAY},
    #                 {'params': net.decoder.parameters(), 'lr':cfg.TRAIN.LR*20, 'weight_decay':cfg.TRAIN.WEIGHT_DECAY}                
    #                 ])

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)

    _t = {'train time' : Timer(),'val time' : Timer()} 

    validate(val_loader, net, criterion, optimizer, -1, restore_transform)
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print 'training time of one epoch: {:.2f}s'.format(_t['train time'].diff)
        _t['val time'].tic()
        validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        _t['val time'].toc(average=False)
        print 'val time of one epoch: {:.2f}s'.format(_t['val time'].diff)


def train(train_loader, net, criterion, optimizer, epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            outputs = outputs[:, :cfg.DATA.NUM_CLASSES - 1, :, :]
            prediction = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
            mean_iu = calculate_mean_iu(prediction, labels.data.cpu().numpy(), cfg.DATA.NUM_CLASSES)
            print '[epoch %d], [iter %d], [training loss %.4f], [mean_iu %.4f]' % (epoch + 1, i + 1, loss.data[0], mean_iu)


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []

    for vi, data in enumerate(val_loader, 0):
        if random.random() > cfg.VAL.SAMPLE_RATE:
            continue
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
    val_loss = criterion(output_batches, label_batches)
    val_loss = val_loss.data[0]

    output_batches = output_batches.cpu().data[:, :cfg.DATA.NUM_CLASSES - 1, :, :]
    label_batches = label_batches.cpu().data.numpy()
    prediction_batches = output_batches.max(1)[1].squeeze_(1).numpy()

    mean_iu = calculate_mean_iu(prediction_batches, label_batches, cfg.DATA.NUM_CLASSES)

    writer.add_scalar('loss', val_loss, epoch + 1)
    writer.add_scalar('mean_iu', mean_iu, epoch + 1)

    if mean_iu > train_record['best_val_mean_iu']:
        train_record['best_val_mean_iu'] = mean_iu
        train_record['corr_epoch'] = epoch + 1
        train_record['corr_loss'] = val_loss
        
        snapshot_name = []
        to_saved_weight = []
        if cfg.TRAIN.STAGE=='encoder':
            snapshot_name = 'encoder_ep_%d_mIoU_%.4f' % (epoch + 1, mean_iu)           
            if len(cfg.TRAIN.GPU_ID)>1:
                to_saved_weight = net.module.encoder.state_dict()                
            else:
                to_saved_weight = net.encoder.state_dict()
        else:
            snapshot_name = 'all_ep_%d_mIoU_%.4f' % (epoch + 1, mean_iu)
            if len(cfg.TRAIN.GPU_ID)>1:
                to_saved_weight = net.module.state_dict()                
            else:
                to_saved_weight = net.state_dict()

        # save model
        torch.save(to_saved_weight, os.path.join(cfg.TRAIN.CKPT_PATH, exp_name, snapshot_name + '.pth'))

        # remove the last best model
        rm_file(os.path.join(cfg.TRAIN.CKPT_PATH, exp_name, train_record['best_model_name'] + '.pth'))
        # update and save the best model
        train_record['best_model_name'] = snapshot_name

        with open(log_txt, 'a') as f:
            f.write(snapshot_name + '\n')

        # show the visualizations
        x = []
        for idx, tensor in enumerate(zip(input_batches, prediction_batches, label_batches)):
            if random.random() > cfg.VIS.SAMPLE_RATE:
                continue
            pil_input = restore(tensor[0])
            pil_output = colorize_mask(tensor[1])
            pil_label = colorize_mask(tensor[2])
            x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')),
                      pil_to_tensor(pil_output.convert('RGB'))])
        x = torch.stack(x, 0)
        x = vutils.make_grid(x, nrow=3, padding=5)
        writer.add_image(exp_name + '_epoch_' + str(epoch+1), (x.numpy()*255).astype(np.uint8).transpose(1,2,0))

    print '--------------------------------------------------------'
    print exp_name
    print '[mean iu %.4f], [val loss %.4f]' % (mean_iu, val_loss)
    print '[best mean iu %.4f], [loss %.4f], [epoch %d]' % (
        train_record['best_val_mean_iu'], train_record['corr_loss'], train_record['corr_epoch'])
    print '--------------------------------------------------------'

    net.train()
    criterion.cuda()


if __name__ == '__main__':
    main()








