import torch
from torch.utils.data import Dataset
import pandas as pd


class MyDataset(Dataset):
    """
    Custom Dataset for CI-VAE.

    Args:
        df (pd.DataFrame): DataFrame containing input features and label(s).
        y_label (list, optional): List of column names to be used as labels (default: ["Y"]).
        mode (str, optional): Either 'train' or 'test' (default: 'train').
        device (torch.device, optional): Device on which to place tensors.
    """
    def __init__(self, df: pd.DataFrame, y_label: list = ["Y"], mode: str = 'train', device=None):
        self.mode = mode
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if mode == 'train':
            if y_label[0] not in df.columns:
                raise ValueError(f"Column {y_label[0]} not found in the dataframe.")
            self.oup = df.loc[:, y_label].values
            self.inp = df.drop(columns=y_label)
        else:
            self.inp = df

        self.x_features = self.inp.columns.tolist()
        self.inp = self.inp.values

    def __len__(self) -> int:
        return self.inp.shape[0]

    def __getitem__(self, idx: int):
        inp_tensor = torch.tensor(self.inp[idx], dtype=torch.float32, device=self.device)
        if self.mode == 'train':
            oup_tensor = torch.tensor(self.oup[idx], dtype=torch.float32, device=self.device)
            return inp_tensor, oup_tensor
        return inp_tensor
