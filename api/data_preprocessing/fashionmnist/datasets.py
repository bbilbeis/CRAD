import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import FashionMNIST

class FashionMNIST_truncated(data.Dataset):
    def __init__(self, root, cache_data_set=None, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__(cache_data_set)

    def __build_truncated_dataset__(self, cache_data_set):
        if cache_data_set is None:
            fmnist_dataobj = FashionMNIST(self.root, train=self.train, transform=self.transform, target_transform=self.target_transform, download=self.download)
        else:
            fmnist_dataobj = cache_data_set
        
        data = fmnist_dataobj.data.numpy()
        target = np.array(fmnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
