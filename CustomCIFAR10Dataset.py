import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from typing import Any, Callable, Optional, Tuple
from PIL import Image

class CustomCIFAR10(CIFAR10):
        
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):

        super(CustomCIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform, train=train, download=download)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target