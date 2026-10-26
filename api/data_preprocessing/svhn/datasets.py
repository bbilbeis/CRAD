import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import SVHN

class SVHN_truncated(data.Dataset):
    def __init__(self, root, cache_data_set=None, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.split = 'train' if train else 'test'
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__(cache_data_set)

    def __build_truncated_dataset__(self, cache_data_set):
        if cache_data_set is None:
            svhn_dataobj = SVHN(self.root, split=self.split, transform=self.transform, target_transform=self.target_transform, download=self.download)
        else:
            svhn_dataobj = cache_data_set
        
        # svhn_dataobj.data has shape (N, 3, 32, 32)
        # svhn_dataobj.labels has shape (N,)
        data = svhn_dataobj.data
        target = np.array(svhn_dataobj.labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        # Transpose from (C, H, W) to (H, W, C) so PIL fromarray works
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
