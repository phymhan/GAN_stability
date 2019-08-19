import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.utils.data as data
import os
from PIL import Image


def get_dataset(name, data_dir, sourcefile='', size=64, lsun_categories=None):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
    ])

    if name == 'image':
        dataset = datasets.ImageFolder(data_dir, transform)
        nlabels = len(dataset.classes)
    elif name == 'imagelist':
        dataset = ImageList(data_dir, sourcefile, transform)
        nlabels = len(dataset.classes)
    elif name == 'npy':
        # Only support normalization for now
        dataset = datasets.DatasetFolder(data_dir, npy_loader, ['npy'])
        nlabels = len(dataset.classes)
    elif name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                   transform=transform)
        nlabels = 10
    elif name == 'lsun':
        if lsun_categories is None:
            lsun_categories = 'train'
        dataset = datasets.LSUN(data_dir, lsun_categories, transform)
        nlabels = len(dataset.classes)
    elif name == 'lsun_class':
        dataset = datasets.LSUNClass(data_dir, transform,
                                     target_transform=(lambda t: 0))
        nlabels = 1
    else:
        raise NotImplemented

    return dataset, nlabels


def npy_loader(path):
    img = np.load(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = img/127.5 - 1.
    elif img.dtype == np.float32:
        img = img * 2 - 1.
    else:
        raise NotImplementedError

    img = torch.Tensor(img)
    if len(img.size()) == 4:
        img.squeeze_(0)

    return img


class ImageList(data.Dataset):
    def __init__(self, root, sourcefile, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        with open(sourcefile, 'r') as f:
            images = []
            labels = []
            for line in f.readlines():
                line = line.rstrip().split()
                images.append(os.path.join(root, line[0]))
                labels.append(int(line[1]))
        self.images = images
        self.labels = labels
        self.classes = list(set(labels))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        lbl = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lbl = self.target_transform(lbl)
        return img, lbl

    def __len__(self):
        return len(self.labels)
