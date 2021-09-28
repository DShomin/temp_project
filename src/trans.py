import albumentations as A
from albumentations.augmentations.crops.transforms import RandomCrop
from albumentations.pytorch import ToTensorV2


def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            A.Resize(128, 128),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(128, 128),
            ToTensorV2(),
        ])

    elif data == 'None':
        return A.Compose([
            A.Resize(128, 128),
            ToTensorV2(),
        ])

    elif data == 'test':
        return A.Compose([
            A.Resize(128, 128),
            ToTensorV2(),
        ])