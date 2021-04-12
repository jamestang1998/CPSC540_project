import torch
import pandas as pd
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class TitanicDataset(Dataset):

    def __init__(self, csvpath, mode='train'):
        self.mode = mode
        df = pd.read_csv(csvpath)
        """       
        <------Some Data Preprocessing---------->
        Removing Null Values, Outliers and Encoding the categorical labels etc
        """
        if self.mode == 'train':
            df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
            df['Ticket'] = LabelEncoder().fit_transform(df['Ticket'])
            df['Cabin'] = LabelEncoder().fit_transform(df['Cabin'])
            df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
            df = df.drop('Name', 1)
            df = df.fillna(0)
            print(df.dtypes)
            self.inp = df.iloc[:, 2:].values
            self.oup = df.iloc[:, 1].values.reshape(891, 1)
        else:
            self.inp = df.values

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt = torch.Tensor(self.inp[idx])
            oupt = torch.Tensor(self.oup[idx])
            return {'inp': inpt, 'oup': oupt}
        else:
            inpt = torch.Tensor(self.inp[idx])
        return {'inp': inpt}