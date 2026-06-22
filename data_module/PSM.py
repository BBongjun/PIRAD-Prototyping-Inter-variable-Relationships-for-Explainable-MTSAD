# import os
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from typing import Optional, Tuple, Any


# class PSMDataset(Dataset):
#     def __init__(
#         self,
#         windows: np.ndarray,
#         labels: np.ndarray,
#         future_windows: Optional[np.ndarray] = None,
#     ):
#         self.windows = windows.astype(np.float32)
#         self.labels = labels.astype(np.float32)
#         self.future_windows = future_windows.astype(np.float32) if future_windows is not None else None
#         self.diagnosis = [[] for _ in range(len(self.labels))]  # placeholder

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx: int):
#         x = torch.tensor(self.windows[idx])
#         y = torch.tensor(self.labels[idx])
#         if self.future_windows is not None:
#             f = torch.tensor(self.future_windows[idx])
#             return x, y, f
#         return x, y


# def create_windows(data: np.ndarray, labels: np.ndarray, window_size: int, forecast: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
#     windows = []
#     targets = []
#     future_windows = [] if forecast else None

#     for idx in range(len(data)):
#         # main window
#         if idx < window_size:
#             pad = data[[0]].repeat(window_size - idx - 1, axis=0)
#             window = np.concatenate([pad, data[:idx + 1]], axis=0)
#         else:
#             window = data[idx - window_size + 1:idx + 1]
#         windows.append(window)
#         targets.append(labels[idx])

#         # forecast window
#         if forecast:
#             if idx + window_size >= len(data):
#                 pad = data[[-1]].repeat(idx + window_size - len(data) + 1, axis=0)
#                 future = np.concatenate([data[idx + 1:], pad], axis=0)
#             else:
#                 future = data[idx + 1:idx + 1 + window_size]
#             future_windows.append(future)

#     windows = np.stack(windows)
#     targets = np.array(targets)
#     if forecast:
#         future_windows = np.stack(future_windows)
#         return windows, targets, future_windows
#     else:
#         return windows, targets, None


# class PSMDataModule:
#     def __init__(
#         self,
#         data_dir: str = "data/PSM",
#         input_size: int = 25,
#         window_size: int = 10,
#         batch_size: int = 64,
#         num_workers: int = 0,
#         pin_memory: bool = False,
#         forecast: bool = False,
#         post_scaler_class: Any = StandardScaler,
#     ):
#         self.data_dir = data_dir
#         self.input_size = input_size
#         self.window_size = window_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.forecast = forecast
#         self.post_scaler_class = post_scaler_class

#         self.train_file = "train.csv"
#         self.test_file = "test.csv"
#         self.test_label_file = "test_label.csv"

#         self.data_train = None
#         self.data_val = None
#         self.data_test = None

#     def setup(self):
#         # 1. Load train data
#         train_path = os.path.join(self.data_dir, self.train_file)
#         train_df = pd.read_csv(train_path, skiprows=1, usecols=range(1, self.input_size + 1)).values.astype(np.float32)

#         # 2. Split
#         split = int(len(train_df) * 0.8)
#         train_raw = train_df[:split]
#         val_raw = train_df[split:]

#         # 3. Fit scaler
#         self.scaler = self.post_scaler_class()
#         self.scaler.fit(train_raw)

#         train_scaled = self.scaler.transform(train_raw)
#         val_scaled = self.scaler.transform(val_raw)

#         # 4. Prepare test
#         test_data = pd.read_csv(os.path.join(self.data_dir, self.test_file), skiprows=1, usecols=range(1, self.input_size + 1)).values.astype(np.float32)
#         test_scaled = self.scaler.transform(test_data)
#         test_labels = pd.read_csv(os.path.join(self.data_dir, self.test_label_file), skiprows=1, usecols=[1]).values.squeeze().astype(np.float32)

#         # 5. Create windows
#         train_windows, train_labels, train_future = create_windows(train_scaled, np.zeros(len(train_scaled)), self.window_size, self.forecast)
#         val_windows, val_labels, val_future = create_windows(val_scaled, np.zeros(len(val_scaled)), self.window_size, self.forecast)
#         test_windows, test_labels_proc, test_future = create_windows(test_scaled, test_labels, self.window_size, self.forecast)

#         # 6. Dataset 구성
#         self.data_train = PSMDataset(train_windows, train_labels, train_future)
#         self.data_val = PSMDataset(val_windows, val_labels, val_future)
#         self.data_test = PSMDataset(test_windows, test_labels_proc, test_future)

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.data_train,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.data_val,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory
#         )

#     def test_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.data_test,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory
#         )

#     def predict_dataloader(self) -> DataLoader:
#         return self.test_dataloader()


# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from typing import Optional, Any, Tuple


# def generate_windows(
#     data: np.ndarray,
#     labels: np.ndarray,
#     window_size: int,
#     forecast: bool = False
# ):
#     windows, targets, forecasts = [], [], []

#     for i in range(len(data)):
#         if i < window_size:
#             pad = data[[0]].repeat(window_size - i - 1, axis=0)
#             window = np.concatenate([pad, data[:i + 1]], axis=0)
#         else:
#             window = data[i - window_size + 1:i + 1]
#         windows.append(window)
#         targets.append(labels[i])

#         if forecast:
#             if i + window_size >= len(data):
#                 pad = data[[-1]].repeat(i + window_size - len(data) + 1, axis=0)
#                 future = np.concatenate([data[i + 1:], pad], axis=0)
#             else:
#                 future = data[i + 1:i + 1 + window_size]
#             forecasts.append(future)

#     return (
#         np.stack(windows),
#         np.array(targets),
#         np.stack(forecasts) if forecast else None
#     )


# class PSMDataset(Dataset):
#     def __init__(
#         self,
#         data: np.ndarray,
#         labels: np.ndarray,
#         future: Optional[np.ndarray] = None
#     ):
#         self.data = data.astype(np.float32)
#         self.labels = labels.astype(np.float32)
#         self.future = future.astype(np.float32) if future is not None else None

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         x = torch.tensor(self.data[idx])
#         y = torch.tensor(self.labels[idx])
#         if self.future is not None:
#             f = torch.tensor(self.future[idx])
#             return x, y, f
#         return x, y


# class PSMDataModule:
#     def __init__(
#         self,
#         data_dir: str = "data/PSM",
#         input_size: int = 25,
#         window_size: int = 10,
#         batch_size: int = 64,
#         num_workers: int = 0,
#         pin_memory: bool = False,
#         forecast: bool = False,
#         post_scaler_class: Any = StandardScaler
#     ):
#         self.data_dir = data_dir
#         self.input_size = input_size
#         self.window_size = window_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.forecast = forecast
#         self.post_scaler_class = post_scaler_class

#         self.train_file = 'train.csv'
#         self.test_file = 'test.csv'
#         self.label_file = 'test_label.csv'

#     def setup(self):
#         # Load train/val
#         train_path = os.path.join(self.data_dir, self.train_file)
        
#         train_raw = np.genfromtxt(train_path, delimiter=',', skip_header=1, usecols=range(1, self.input_size + 1), filling_values=0).astype(np.float32)
#         split = int(len(train_raw) * 0.8)
#         train_data, val_data = train_raw[:split], train_raw[split:]

#         self.scaler = self.post_scaler_class()
#         self.scaler.fit(train_data)
#         train_scaled = self.scaler.transform(train_data)
#         val_scaled = self.scaler.transform(val_data)

#         train_x, train_y, train_f = generate_windows(train_scaled, np.zeros(len(train_scaled)), self.window_size, self.forecast)
#         val_x, val_y, val_f = generate_windows(val_scaled, np.zeros(len(val_scaled)), self.window_size, self.forecast)

#         # Load test
#         test_path = os.path.join(self.data_dir, self.test_file)
#         label_path = os.path.join(self.data_dir, self.label_file)

#         test_raw = np.genfromtxt(test_path, delimiter=',', skip_header=1, usecols=range(1, self.input_size + 1), filling_values=0).astype(np.float32)
#         test_labels = np.loadtxt(label_path, delimiter=',', skiprows=1, usecols=1).astype(np.float32)

#         test_scaled = self.scaler.transform(test_raw)
#         test_x, test_y, test_f = generate_windows(test_scaled, test_labels, self.window_size, self.forecast)

#         self.data_train = PSMDataset(train_x, train_y, train_f)
#         self.data_val = PSMDataset(val_x, val_y, val_f)
#         self.data_test = PSMDataset(test_x, test_y, test_f)

#     def train_dataloader(self):
#         return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def val_dataloader(self):
#         return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def test_dataloader(self):
#         return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def predict_dataloader(self):
#         return self.test_dataloader()



import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Optional, Any


def generate_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    forecast: bool = False
):
    windows, targets, forecasts = [], [], []

    for i in range(len(data)):
        if i < window_size:
            pad = data[[0]].repeat(window_size - i - 1, axis=0)
            window = np.concatenate([pad, data[:i + 1]], axis=0)
        else:
            window = data[i - window_size + 1:i + 1]
        windows.append(window)
        targets.append(labels[i])

        if forecast:
            if i + window_size >= len(data):
                pad = data[[-1]].repeat(i + window_size - len(data) + 1, axis=0)
                future = np.concatenate([data[i + 1:], pad], axis=0)
            else:
                future = data[i + 1:i + 1 + window_size]
            forecasts.append(future)

    return (
        np.stack(windows),
        np.array(targets),
        np.stack(forecasts) if forecast else None
    )


class PSMDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        future: Optional[np.ndarray] = None
    ):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.future = future.astype(np.float32) if future is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        y = torch.tensor(self.labels[idx])
        if self.future is not None:
            f = torch.tensor(self.future[idx])
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
        use_scaler: bool = True,               # ? StandardScaler 적용 여부
        post_scaler_class: Any = StandardScaler
    ):
        self.data_dir = data_dir
        self.input_size = input_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.forecast = forecast
        self.use_scaler = use_scaler
        self.post_scaler_class = post_scaler_class

        self.train_file = 'train.csv'
        self.test_file = 'test.csv'
        self.label_file = 'test_label.csv'

    def setup(self):
        # Load train/val
        train_path = os.path.join(self.data_dir, self.train_file)
        train_raw = np.genfromtxt(train_path, delimiter=',', skip_header=1,
                                  usecols=range(1, self.input_size + 1),
                                  filling_values=0).astype(np.float32)
        
        split = int(len(train_raw) * 0.8)
        train_data, val_data = train_raw[:split], train_raw[split:]

        # Scaling (use_scaler 플래그로 제어)
        if self.use_scaler:
            self.scaler = self.post_scaler_class()
            self.scaler.fit(train_data)
            train_scaled = self.scaler.transform(train_data)
            val_scaled = self.scaler.transform(val_data)
        else:
            self.scaler = None
            train_scaled = train_data.astype(np.float32)
            val_scaled = val_data.astype(np.float32)

        train_x, train_y, train_f = generate_windows(train_scaled, np.zeros(len(train_scaled)),
                                                     self.window_size, self.forecast)
        val_x, val_y, val_f = generate_windows(val_scaled, np.zeros(len(val_scaled)),
                                               self.window_size, self.forecast)

        # Load test
        test_path = os.path.join(self.data_dir, self.test_file)
        label_path = os.path.join(self.data_dir, self.label_file)

        test_raw = np.genfromtxt(test_path, delimiter=',', skip_header=1,
                                 usecols=range(1, self.input_size + 1),
                                 filling_values=0).astype(np.float32)
        test_labels = np.loadtxt(label_path, delimiter=',', skiprows=1, usecols=1).astype(np.float32)

        if self.use_scaler:
            test_scaled = self.scaler.transform(test_raw)
        else:
            test_scaled = test_raw.astype(np.float32)

        test_x, test_y, test_f = generate_windows(test_scaled, test_labels,
                                                  self.window_size, self.forecast)

        self.data_train = PSMDataset(train_x, train_y, train_f)
        self.data_val = PSMDataset(val_x, val_y, val_f)
        self.data_test = PSMDataset(test_x, test_y, test_f)

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
