import argparse
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pdb
from PIL import Image
import numpy as np
import os


# TODO 

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDataPath', type=str, default='/mount/gjy/Dataset/City/train', 
                        help='absolute path to your data path')
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()

    imgs_list = []

    # for i_img, img_name in enumerate(os.listdir(args.trainDataPath)):
    # 	if i_img % 100 == 0:
    #         print i_img
    #     img = np.array(Image.open(os.path.join(args.trainDataPath, img_name)))
    #     imgs_list.append(img)

    # pdb.set_trace()
    # imgs = np.array(imgs_list).astype(np.float32)/255.

    data = dset.ImageFolder(args.trainDataPath, transform=transforms.ToTensor())
    print(data.root)
    pdb.set_trace()
    imgs = np.array(data)
    
    means = []
    stdevs = []
    for i in range(3):
        pixels = imgs[:,i,:,:].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))
    print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))