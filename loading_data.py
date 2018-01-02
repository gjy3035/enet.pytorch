import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import transforms as own_transforms
from cityscapes import CityScapes
from config import cfg

def loading_data():
    mean_std = cfg.DATA.MEAN_STD
    train_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
        own_transforms.RandomHorizontallyFlip()
    ])
    val_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.CenterCrop(cfg.TRAIN.IMG_SIZE)
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = standard_transforms.Compose([
        own_transforms.MaskToTensor(),
        own_transforms.ChangeLabel(cfg.DATA.IGNORE_LABEL, cfg.DATA.NUM_CLASSES - 1)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = CityScapes('train', simul_transform=train_simul_transform, transform=img_transform,
                           target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=16, shuffle=True)
    val_set = CityScapes('val', simul_transform=val_simul_transform, transform=img_transform,
                         target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=cfg.VAL.BATCH_SIZE, num_workers=16, shuffle=False)

    return train_loader, val_loader, restore_transform
