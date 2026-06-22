# import os
# from typing import Any, Optional, Tuple
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler


# def parse_diagnosis(file_path: str) -> dict:
#     parsed_data = {}
#     with open(file_path, 'r') as file:
#         for line in file:
#             range_part, numbers_part = line.strip().split(':')
#             start, end = map(int, range_part.split('-'))
#             numbers = [int(num) - 1 for num in numbers_part.split(',')]
#             parsed_data[(start, end)] = numbers
#     return parsed_data


# def generate_windows(data: np.ndarray, labels: np.ndarray, window_size: int, forecast: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
#     windows = []
#     target_labels = []
#     forecast_windows = [] if forecast else None

#     for idx in range(len(data)):
#         if idx < window_size:
#             pad = data[[0]].repeat(window_size - idx - 1, axis=0)
#             window = np.concatenate([pad, data[:idx + 1]], axis=0)
#         else:
#             window = data[idx - window_size + 1:idx + 1]

#         windows.append(window)
#         target_labels.append(labels[idx])

#         if forecast:
#             if idx + window_size >= len(data):
#                 pad = data[[-1]].repeat(idx + window_size - len(data) + 1, axis=0)
#                 future = np.concatenate([data[idx + 1:], pad], axis=0)
#             else:
#                 future = data[idx + 1:idx + 1 + window_size]
#             forecast_windows.append(future)

#     windows = np.stack(windows)
#     target_labels = np.array(target_labels)

#     if forecast:
#         forecast_windows = np.stack(forecast_windows)
#         return windows, target_labels, forecast_windows
#     return windows, target_labels, None


# class SMDDataset(Dataset):
#     def __init__(
#         self,
#         windows: np.ndarray,
#         labels: np.ndarray,
#         future_windows: Optional[np.ndarray] = None,
#         diagnosis_path: Optional[str] = None,
#     ):
#         self.windows = windows.astype(np.float32)
#         self.labels = labels.astype(np.float32)
#         self.future_windows = future_windows.astype(np.float32) if future_windows is not None else None
#         self.diagnosis = [[] for _ in range(len(self.labels))]

#         if diagnosis_path and os.path.exists(diagnosis_path):
#             diagnosis_data = parse_diagnosis(diagnosis_path)
#             for i in range(len(self.labels)):
#                 for (start, end), v in diagnosis_data.items():
#                     if start <= i < end:
#                         self.diagnosis[i] = v

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx: int):
#         x = torch.tensor(self.windows[idx])
#         y = torch.tensor(self.labels[idx])
#         if self.future_windows is not None:
#             f = torch.tensor(self.future_windows[idx])
#             return x, y, f
#         return x, y


# class SMDDataModule:
#     def __init__(
#         self,
#         machine_id: str = "machine-1-1",
#         data_dir: str = "data/SMD",
#         input_size: int = 38,
#         window_size: int = 10,
#         batch_size: int = 64,
#         num_workers: int = 0,
#         pin_memory: bool = False,
#         post_scaler_class: Any = StandardScaler,
#         forecast: bool = False,
#     ) -> None:
#         self.machine_id = machine_id
#         self.data_dir = data_dir
#         self.input_size = input_size
#         self.window_size = window_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.post_scaler_class = post_scaler_class
#         self.forecast = forecast

#         self.train_file = f"train/{machine_id}.txt"
#         self.test_file = f"test/{machine_id}.txt"
#         self.test_label_file = f"test_label/{machine_id}.txt"
#         self.diagnosis_file = f"interpretation_label/{machine_id}.txt"

#         self.data_train = None
#         self.data_val = None
#         self.data_test = None

#     def setup(self):
#         if self.data_train and self.data_val and self.data_test:
#             return

#         # 1. Load & scale
#         train_data = np.loadtxt(os.path.join(self.data_dir, self.train_file), delimiter=',', dtype=np.float32)
#         split = int(len(train_data) * 0.8)
#         train_raw = train_data[:split]
#         val_raw = train_data[split:]

#         self.scaler = self.post_scaler_class()
#         self.scaler.fit(train_raw)
#         train_scaled = self.scaler.transform(train_raw)
#         val_scaled = self.scaler.transform(val_raw)

#         # 2. Generate windows
#         train_x, train_y, train_f = generate_windows(train_scaled, np.zeros(len(train_scaled)), self.window_size, self.forecast)
#         val_x, val_y, val_f = generate_windows(val_scaled, np.zeros(len(val_scaled)), self.window_size, self.forecast)

#         # 3. Test
#         test_data = np.loadtxt(os.path.join(self.data_dir, self.test_file), delimiter=',', dtype=np.float32)
#         test_scaled = self.scaler.transform(test_data)
#         test_labels = np.loadtxt(os.path.join(self.data_dir, self.test_label_file), delimiter=',', usecols=0, dtype=np.float32)
#         test_x, test_y, test_f = generate_windows(test_scaled, test_labels, self.window_size, self.forecast)

#         diagnosis_path = os.path.join(self.data_dir, self.diagnosis_file)

#         # 4. Datasets
#         self.data_train = SMDDataset(train_x, train_y, train_f)
#         self.data_val = SMDDataset(val_x, val_y, val_f)
#         self.data_test = SMDDataset(test_x, test_y, test_f, diagnosis_path=diagnosis_path)

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def test_dataloader(self) -> DataLoader:
#         return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def predict_dataloader(self) -> DataLoader:
#         return self.test_dataloader()

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


def generate_windows(data: np.ndarray, labels: np.ndarray, window_size: int, forecast: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    windows = []
    target_labels = []
    forecast_windows = [] if forecast else None

    for idx in range(len(data)):
        if idx < window_size:
            pad = data[[0]].repeat(window_size - idx - 1, axis=0)
            window = np.concatenate([pad, data[:idx + 1]], axis=0)
        else:
            window = data[idx - window_size + 1:idx + 1]

        windows.append(window)
        target_labels.append(labels[idx])

        if forecast:
            if idx + window_size >= len(data):
                pad = data[[-1]].repeat(idx + window_size - len(data) + 1, axis=0)
                future = np.concatenate([data[idx + 1:], pad], axis=0)
            else:
                future = data[idx + 1:idx + 1 + window_size]
            forecast_windows.append(future)

    windows = np.stack(windows)
    target_labels = np.array(target_labels)

    if forecast:
        forecast_windows = np.stack(forecast_windows)
        return windows, target_labels, forecast_windows
    return windows, target_labels, None


class SMDDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        future_windows: Optional[np.ndarray] = None,
        diagnosis_path: Optional[str] = None,
    ):
        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.future_windows = future_windows.astype(np.float32) if future_windows is not None else None
        self.diagnosis = [[] for _ in range(len(self.labels))]

        if diagnosis_path and os.path.exists(diagnosis_path):
            diagnosis_data = parse_diagnosis(diagnosis_path)
            for i in range(len(self.labels)):
                for (start, end), v in diagnosis_data.items():
                    if start <= i < end:
                        self.diagnosis[i] = v

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.windows[idx])
        y = torch.tensor(self.labels[idx])
        if self.future_windows is not None:
            f = torch.tensor(self.future_windows[idx])
            return x, y, f
        return x, y


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
        forecast: bool = False,
        use_scaler: bool = True,             # ? StandardScaler 적용 여부
        post_scaler_class: Any = StandardScaler,
    ) -> None:
        self.machine_id = machine_id
        self.data_dir = data_dir
        self.input_size = input_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.forecast = forecast
        self.use_scaler = use_scaler
        self.post_scaler_class = post_scaler_class

        self.train_file = f"train/{machine_id}.txt"
        self.test_file = f"test/{machine_id}.txt"
        self.test_label_file = f"test_label/{machine_id}.txt"
        self.diagnosis_file = f"interpretation_label/{machine_id}.txt"

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self):
        if self.data_train and self.data_val and self.data_test:
            return

        # 1. Load raw data
        train_data = np.loadtxt(os.path.join(self.data_dir, self.train_file), delimiter=',', dtype=np.float32)
        split = int(len(train_data) * 0.8)
        train_raw = train_data[:split]
        val_raw = train_data[split:]

        # 2. Scaling (use_scaler 플래그로 제어)
        if self.use_scaler:
            self.scaler = self.post_scaler_class()
            self.scaler.fit(train_raw)
            train_scaled = self.scaler.transform(train_raw)
            val_scaled = self.scaler.transform(val_raw)
        else:
            self.scaler = None
            train_scaled = train_raw.astype(np.float32)
            val_scaled = val_raw.astype(np.float32)

        # 3. Train/Val windows
        train_x, train_y, train_f = generate_windows(train_scaled, np.zeros(len(train_scaled)), self.window_size, self.forecast)
        val_x, val_y, val_f = generate_windows(val_scaled, np.zeros(len(val_scaled)), self.window_size, self.forecast)

        # 4. Test
        test_data = np.loadtxt(os.path.join(self.data_dir, self.test_file), delimiter=',', dtype=np.float32)
        if self.use_scaler:
            test_scaled = self.scaler.transform(test_data)
        else:
            test_scaled = test_data.astype(np.float32)

        test_labels = np.loadtxt(os.path.join(self.data_dir, self.test_label_file), delimiter=',', usecols=0, dtype=np.float32)
        test_x, test_y, test_f = generate_windows(test_scaled, test_labels, self.window_size, self.forecast)

        diagnosis_path = os.path.join(self.data_dir, self.diagnosis_file)

        # 5. Datasets
        self.data_train = SMDDataset(train_x, train_y, train_f)
        self.data_val = SMDDataset(val_x, val_y, val_f)
        self.data_test = SMDDataset(test_x, test_y, test_f, diagnosis_path=diagnosis_path)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
