import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MNISTDataset(Dataset):
    
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    
    
    

    def __init__(self, root, train, transform=None):
        self.train = train
        self.processed_folder = 'processed'
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
            
        self.root_dir = root

        self.data, self.targets = torch.load(os.path.join(root, self.processed_folder, data_file))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return index, img, target
