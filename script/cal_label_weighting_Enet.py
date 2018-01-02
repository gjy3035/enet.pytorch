#!/usr/bin/env python
# -*- coding: utf-8 -*
"""
The code is based on https://github.com/TimoSaemann/ENet/blob/master/scripts/calculate_class_weighting.py.
This script calculates the class weighting for the "SoftmaxWithLoss" layer.
cf. https://arxiv.org/pdf/1411.4734.pdf
"we weight each pixel by ¦Ác = median freq/freq(c) where freq(c) is the number of pixels of class c divided by the total
number of pixels in images where c is present, and median freq is the median of these frequencies."
"""
import argparse
import numpy as np
import os
from PIL import Image
import pdb

__author__ = 'GJY' 
__email__ = 'gjy3035@gmail.com'
__data__ = '18th Dec, 2017'

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDataPath', type=str, default='/home/optimal/GJY/City/train/mask', help='absolute path to your data path')
    parser.add_argument('--num_classes', type=int, default=19, help='absolute path to your data file')
    return parser


if __name__ == '__main__':

    args = make_parser().parse_args()

    classes, freq, class_weights, present_in_data, a = ([0 for i in xrange(args.num_classes)] for i in xrange(5))
    image_nr = 0
    median_freq = 0

    for i_img, img_name in enumerate(os.listdir(args.trainDataPath)):

        labels = np.array(Image.open(os.path.join(args.trainDataPath, img_name)))
        # pdb.set_trace()
        if i_img % 100 == 0:
            print i_img

        for i in xrange(args.num_classes):
            if (np.sum((labels == i))) == 0:    
                pass
            else:
                classes[i] += (labels == i).sum()  # sum up all pixels that belongs to a certain class
                present_in_data[i] += 1  # how often the class is present in the dataset

    pdb.set_trace()
    classes = np.array(classes).astype(np.float64)
    norm_classes = classes / classes.sum()
    weight = 1 / np.log(norm_classes + 1.02)

    for m in xrange(args.num_classes):
        print '    class_weighting: {:.4f}'.format(weight[m])

    print weight

    print "Done!"

