from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch
import random

class AddGaussianNoise(object):
        def __init__(self, mean=0.0, std=0.1):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return tensor + noise

def create_dataloader(batch_size, shuffle=True):
    # Function to add Gaussian noise
    

    # Define transformations with augmentation
    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),  # 50% chance of flipping
        T.RandomRotation(10),  # Rotate randomly by ±10 degrees
        # T.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop
        T.ToTensor(),
        # AddGaussianNoise(0.0, 0.05),  # Apply Gaussian noise
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Randomly erase a patch
    ])

    # Validation set does NOT require augmentation, only normalization
    transform_val = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    root_train = 'dataset/tiny-imagenet/tiny-imagenet-200/train'
    root_val = 'dataset/tiny-imagenet/tiny-imagenet-200/val'
    
    tiny_imagenet_dataset_train = ImageFolder(root=root_train, transform=transform_train)
    tiny_imagenet_dataset_val = ImageFolder(root=root_val, transform=transform_val)
    
    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader