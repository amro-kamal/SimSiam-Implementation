from torchvision.transforms import transforms
import torchvision
import torch

def simsiam_cifar10_loader(batch_size):
  
    train_data = torchvision.datasets.CIFAR10('data/train',train=True,download=True, transform=Transform())
    val_data = torchvision.datasets.CIFAR10('data/val',train=False,download=True, transform=Transform())

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,num_workers=4)

    return train_loader, val_loader

def cifar10_loader(batch_size):
    train_transforms = transforms.Compose([
                                        transforms.RandomResizedCrop(32) ,
                                        transforms.ToTensor()  ,
                                        transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    val_transforms = transforms.Compose([
                                        transforms.Resize(32) ,
                                        transforms.ToTensor()  ,
                                        transforms.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])
    train_data = torchvision.datasets.CIFAR10('data/train',train=True,download=True, transform=train_transforms)
    val_data = torchvision.datasets.CIFAR10('data/val',train=False,download=True, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,num_workers=4)

    return train_loader, val_loader

class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2,1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2,1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


