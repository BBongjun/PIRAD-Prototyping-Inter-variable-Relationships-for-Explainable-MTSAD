# data_module/smd_dataloader.py

import os
from typing import Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def parse_diagnosis(file_path: str) -> dict:
    parsed_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            range_part, numbers_part = line.strip().split(':')
            start, end = map(int, range_part.split('-'))
            numbers = [int(num) - 1 for num in numbers_part.split(',')]
            parsed_data[(start, end)] = numbers
    return parsed_data


class SMDDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        forecast: bool = False,
        diagnosis_path: Optional[str] = None,
    ):
        super().__init__()

        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.forecast = forecast
        
        self.diagnosis = [[] for _ in range(self.data.shape[0])]
        if diagnosis_path and os.path.exists(diagnosis_path):
            diagnosis_data = parse_diagnosis(diagnosis_path)
            for i in range(self.data.shape[0]):
                for (start, end), v in diagnosis_data.items():
                    if start <= i < end:
                        self.diagnosis[i] = v

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < self.window_size:
            start = self.data[[0]].repeat(self.window_size - idx - 1, axis=0)
            window = np.concatenate((start, self.data[:idx + 1]), axis=0)
        else:
            window = self.data[idx - self.window_size + 1:idx + 1]

        if not self.forecast:
            return torch.tensor(window, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

        if idx > len(self) - self.window_size - 1:
            end = self.data[[-1]].repeat(self.window_size + idx - len(self) + 1, axis=0)
            forecast_window = np.concatenate((self.data[idx + 1:], end), axis=0)
        else:
            forecast_window = self.data[idx + 1: idx + 1 + self.window_size]

        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(forecast_window, dtype=torch.float32),
        )


class SMDDataModule:
    def __init__(
        self,
        machine_id: str = "machine-1-1",
        data_dir: str = "data/SMD",
        input_size: int = 38,
        window_size: int = 10,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        post_scaler_class: Any = StandardScaler,
        forecast: bool = False,
    ) -> None:
        self.machine_id = machine_id
        self.data_dir = data_dir
        self.input_size = input_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.post_scaler_class = post_scaler_class
        self.forecast = forecast

        # 머신 ID 기반으로 파일 경로 자동 설정
        self.train_file = f"train/{machine_id}.txt"
        self.test_file = f"test/{machine_id}.txt"
        self.test_label_file = f"test_label/{machine_id}.txt"
        self.diagnosis_file = f"interpretation_label/{machine_id}.txt"

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self):
        if self.data_train and self.data_val and self.data_test:
            return

        # 1. Load raw train data
        raw_train_data = np.loadtxt(os.path.join(self.data_dir, self.train_file), delimiter=',', dtype=np.float32)
        total_len = raw_train_data.shape[0]
        train_len = int(total_len * 0.8)
        val_len = total_len - train_len

        # 2. Split by order (시계열 순서 유지)
        train_data = raw_train_data[:train_len]
        val_data = raw_train_data[train_len:]

        # 3. Fit scaler only on train
        self.scaler = self.post_scaler_class()
        self.scaler.fit(train_data)

        # 4. Apply transform
        train_scaled = self.scaler.transform(train_data)
        val_scaled = self.scaler.transform(val_data)

        # 5. Load test data & apply same scaler
        test_data = np.loadtxt(os.path.join(self.data_dir, self.test_file), delimiter=',', dtype=np.float32)
        test_scaled = self.scaler.transform(test_data)
        test_labels = np.loadtxt(os.path.join(self.data_dir, self.test_label_file), delimiter=',', usecols=0, dtype=np.float32)
        diagnosis_path = os.path.join(self.data_dir, self.diagnosis_file)

        # 6. Create Datasets
        self.data_train = SMDDataset(train_scaled, np.zeros(train_len), window_size=self.window_size, forecast=self.forecast)
        self.data_val = SMDDataset(val_scaled, np.zeros(val_len), window_size=self.window_size, forecast=self.forecast)
        self.data_test = SMDDataset(test_scaled, test_labels, window_size=self.window_size, forecast=self.forecast, diagnosis_path=diagnosis_path)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.window_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )