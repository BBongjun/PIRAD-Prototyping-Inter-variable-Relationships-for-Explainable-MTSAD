import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Optional, Any, Tuple


class PSMDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        forecast: bool = False
    ):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.window_size = window_size
        self.forecast = forecast

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # ? Padding if not enough previous context
        if idx < self.window_size:
            pad = self.data[[0]].repeat(self.window_size - idx - 1, axis=0)
            window = np.concatenate([pad, self.data[:idx + 1]], axis=0)
        else:
            window = self.data[idx - self.window_size + 1:idx + 1]
 
        x = torch.tensor(window)
        y = torch.tensor(self.labels[idx])

        if self.forecast:
            start_f = idx + 1
            end_f = start_f + self.window_size
            if end_f > len(self.data):
                future = np.concatenate([
                    self.data[start_f:],
                    self.data[[-1]].repeat(end_f - len(self.data), axis=0)
                ], axis=0)
            else:
                future = self.data[start_f:end_f]
            f = torch.tensor(future)
            return x, y, f

        return x, y


class PSMDataModule:
    def __init__(
        self,
        data_dir: str = "data/PSM",
        input_size: int = 25,
        window_size: int = 10,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        forecast: bool = False,
        post_scaler_class: Any = StandardScaler
    ):
        self.data_dir = data_dir
        self.input_size = input_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.forecast = forecast
        self.post_scaler_class = post_scaler_class

        self.train_file = 'train.csv'
        self.test_file = 'test.csv'
        self.label_file = 'test_label.csv'

    def setup(self):
        # Load train/val
        train_path = os.path.join(self.data_dir, self.train_file)
        train_raw = np.genfromtxt(train_path, delimiter=',', skip_header=1,
                                  usecols=range(1, self.input_size + 1), filling_values=0).astype(np.float32)
        split = int(len(train_raw) * 0.8)
        train_data, val_data = train_raw[:split], train_raw[split:]

        self.scaler = self.post_scaler_class()
        self.scaler.fit(train_data)
        train_scaled = self.scaler.transform(train_data)
        val_scaled = self.scaler.transform(val_data)

        self.data_train = PSMDataset(train_scaled, np.zeros(len(train_scaled)), self.window_size, self.forecast)
        self.data_val = PSMDataset(val_scaled, np.zeros(len(val_scaled)), self.window_size, self.forecast)

        # Load test
        test_path = os.path.join(self.data_dir, self.test_file)
        label_path = os.path.join(self.data_dir, self.label_file)

        test_raw = np.genfromtxt(test_path, delimiter=',', skip_header=1,
                                 usecols=range(1, self.input_size + 1), filling_values=0).astype(np.float32)
        test_labels = np.loadtxt(label_path, delimiter=',', skiprows=1, usecols=1).astype(np.float32)

        test_scaled = self.scaler.transform(test_raw)
        self.data_test = PSMDataset(test_scaled, test_labels, self.window_size, self.forecast)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return self.test_dataloader()