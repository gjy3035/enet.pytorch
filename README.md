# ENet.PyTorch
This repository is a PyTorch implementation of [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147). 
The partial codes are based on 
- The official [**Torch** code](https://github.com/e-lab/ENet-training) by the [authors](https://github.com/e-lab) of the paper;
- The unofficial[ **Caffe** code](https://github.com/TimoSaemann/ENet) by [TimoSaemann](https://github.com/TimoSaemann);
- The unofficial [**PyTorch** code](https://github.com/vietdoan/Enet_Pytorch) by [vietdoan](https://github.com/vietdoan);
- [pytorch-semantic-segmentation](https://github.com/ZijunDeng/pytorch-semantic-segmentation) by [ZijunDeng](https://github.com/ZijunDeng).

## Requirement
- PyTorch
- TensorboardX

## Performance
### Quantitative Redults
|   | mean IoU | Model Size|
|------|:------:|:------:|
| encoder     | 53.66% | 1.38M|
| encoder + decoder (step-by-step training) | 56.04% | 1.49M|

The best mean IoU of **56.04%** on the val set is close to the **58.3%** in the paper (58.3% is reported on the test set)

### Qualitative  Redults
|   | exemplars | 
|------|:------:|
| encoder     |![Original images, ground truth, predicted images][1] | 
| encoder + decoder (step-by-step training) | ![Original images, ground truth, predicted images][2] | 

## Usage

### Data Preparation

1. Downloading the [CityScapes dataset](https://www.cityscapes-dataset.com/).
2. Unzip the ```leftImg8bit_trainvaltest.zip``` and the ```gtFine_trainvaltest.zip``` in a certain folder (```root path```).
3. Preprocess the dataset: 
	1). modify the ```root path``` and the ```processed the path``` in ```./script/preprocess.py```;
	2). run ```python ./script/preprocess.py```.
4. Calculate the label weights: ``` python ./script/cal_label_weighting_Enet.py --trainDataPath=your_label_path --num_classes=19```, and repalce the ```__C.TRAIN.LABEL_WEIGHT``` in ```config.py```

### Train model

#### Train the encoder
1. modify config.py:
	```
	__C.TRAIN.STAGE = 'encoder' # encoder or all
	__C.TRAIN.PRETRAINED_ENCODER = '' # 
	```
2. ```python train.py```
	The loss curve line:
	![enter description here][3]
3. Here, we provide the trained [Pytorch model](./ckpt/encoder_ep_497_mIoU_0.5098.pth).

#### Train the ENet based on the encoder
1. modify config.py:
	```
	__C.TRAIN.STAGE = 'all' # encoder or all
	__C.TRAIN.PRETRAINED_ENCODER = './ckpt/encoder_ep_497_mIoU_0.5098.pth' # Path of the pretrained encoder
	```
2. ```python train.py```
	The loss curve line:
	![enter description here][4]
3. Here, we provide the trained [Pytorch model](./ckpt/all_ep_1219_mIoU_0.5324.pth).


  [1]: ./images/1514871757088.jpg
  [2]: ./images/1514871870015.jpg
  [3]: ./images/1514873029686.jpg
  [4]: ./images/1514873238074.jpg