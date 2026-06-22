# import os
# from datetime import datetime
# from typing import Any, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from sklearn.base import TransformerMixin
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import Dataset, DataLoader

# from data_module.components import swat

# class SWaTDataset(Dataset):
#     def __init__(
#         self,
#         data_path: str,
#         attacks: list[dict],
#         input_size: int,
#         window_size: int,
#         post_scaler: Optional[TransformerMixin] = None,
#         post_scaler_class: Any = StandardScaler,
#         max_rows: Optional[int] = None,
#         forecast: bool = False,
#     ) -> None:
#         super().__init__()

#         self.window_size = window_size
#         self.forecast = forecast
        
#         # 1. Load CSV with proper separator and timestamp handling
#         if data_path == 'data/Swat/SWaT_Dataset_Attack_v0.csv':
#             df = pd.read_csv(data_path, low_memory=False, sep=';', nrows=max_rows)
#         else:
#             df = pd.read_csv(data_path, low_memory=False, nrows=max_rows)
#         df['Timestamp'] = pd.to_datetime(df['Timestamp'].str.strip(), format="%d/%m/%Y %I:%M:%S %p")
#         ts = pd.to_datetime(df['Timestamp'].values).to_pydatetime()
#         columns = [col.strip() for col in df.columns if col not in ['Timestamp', 'Normal/Attack']]

#         # 2. Handle labels (Normal → 0, Attack → 1)
#         if 'Normal/Attack' in df.columns:
#             labels = df['Normal/Attack'].astype(str).apply(lambda x: float(x.strip().lower() != 'normal')).values
#             df = df.drop(columns=['Normal/Attack'])
#         else:
#             labels = np.zeros(len(df), dtype=np.float32)

#         # 3. Replace commas and convert to float
#         for col in columns:
#             df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

#         self.data = df[columns].values
#         self.labels = labels
        
#         # 4. Diagnosis marking / generate labels
#         self.diagnosis = []
#         for i, t in enumerate(ts[:len(self.data)]):
#             d = []
#             for attack in attacks:
#                 if attack['start_time_dt'] <= t <= attack['end_time_dt']:
#                     self.labels[i] = 1.0
#                     points = map(lambda p: p.replace('-', ''), attack['points'])
#                     d = [columns.index(p) for p in points if p in columns]
#                     break
#             self.diagnosis.append(d)

#         # 5. Scaling
#         if post_scaler is None:
#             self.post_scaler = post_scaler_class()
#             self.post_scaler.fit(self.data)
#         else:
#             self.post_scaler = post_scaler

#         self.data = self.post_scaler.transform(self.data)

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         if idx < self.window_size:
#             start = self.data[[0]].repeat(self.window_size - idx - 1, axis=0)
#             window = np.concatenate((start, self.data[:idx + 1]), axis=0)
#         else:
#             window = self.data[idx - self.window_size + 1:idx + 1]

#         if not self.forecast:
#             return torch.tensor(window, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

#         if idx > len(self) - self.window_size - 1:
#             end = self.data[[-1]].repeat(self.window_size + idx - len(self) + 1, axis=0)
#             forecast_window = np.concatenate((self.data[idx + 1:], end), axis=0)
#         else:
#             forecast_window = self.data[idx + 1: idx + 1 + self.window_size]

#         return (
#             torch.tensor(window, dtype=torch.float32),
#             torch.tensor(self.labels[idx], dtype=torch.float32),
#             torch.tensor(forecast_window, dtype=torch.float32),
#         )

# class SWaTDataModule:
#     def __init__(
#         self,
#         data_dir: str = "data/",
#         input_size: int = 51,
#         window_size: int = 10,
#         batch_size: int = 64,
#         num_workers: int = 0,
#         pin_memory: bool = False,
#         post_scaler_class: Any = StandardScaler,
#         forecast: bool = False,
#     ) -> None:
#         self.data_dir = os.path.join(data_dir, 'Swat')
#         self.input_size = input_size
#         self.window_size = window_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.post_scaler_class = post_scaler_class
#         self.forecast = forecast

#         self.normal_file = 'SWaT_Dataset_Normal_v1.csv'
#         self.attack_file = 'SWaT_Dataset_Attack_v0.csv'

#         self.data_train: Optional[Dataset] = None
#         self.data_val: Optional[Dataset] = None
#         self.data_test: Optional[Dataset] = None

#     def setup(self):
#         if self.data_train and self.data_val and self.data_test:
#             return

#         data_train = SWaTDataset(
#             os.path.join(self.data_dir, self.normal_file),
#             [],
#             self.input_size,
#             self.window_size,
#             post_scaler_class=self.post_scaler_class,
#             forecast=self.forecast,
#         )

#         self.data_test = SWaTDataset(
#             os.path.join(self.data_dir, self.attack_file),
#             swat.attacks,
#             self.input_size,
#             self.window_size,
#             post_scaler_class=self.data_train.post_scaler,
#             forecast=self.forecast,
#         )

#         self.data_train, self.data_val = train_test_split(data_train, train_size=0.8, shuffle=False)

#     def train_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.data_train,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=True,
#         )

#     def val_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.data_val,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=False,
#         )

#     def test_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.data_test,
#             batch_size=self.window_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=False,
#         )

#     def predict_dataloader(self) -> DataLoader:
#         return DataLoader(
#             self.data_test,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=False,
#         )


# data_module/swat_data_module.py

import os
import pandas as pd
import numpy as np
import torch

from datetime import datetime
from typing import Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from data_module.components import swat


class SWaTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        attacks: Optional[list[dict]],
        columns: list[str],
        window_size: int,
        forecast: bool = False,
        post_scaler_class: Any = StandardScaler,
        post_scaler: Optional[Any] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.forecast = forecast
        self.columns = columns

        # Timestamp, Labels, Features
        self.timestamps = pd.to_datetime(df['Timestamp'].str.strip(), format="%d/%m/%Y %I:%M:%S %p")
        if 'Normal/Attack' in df.columns:
            labels = df['Normal/Attack'].astype(str).apply(lambda x: float(x.strip().lower() != 'normal')).values
            df = df.drop(columns=['Normal/Attack'])
        else:
            labels = np.zeros(len(df), dtype=np.float32)
            
        self.data = df[columns].values
        self.labels = labels

        # Apply attack labels and diagnosis
        self.diagnosis = []
        if attacks:
            for i, t in enumerate(self.timestamps):
                d = []
                for attack in attacks:
                    if attack['start_time_dt'] <= t <= attack['end_time_dt']:
                        self.labels[i] = 1.0
                        points = map(lambda p: p.replace('-', ''), attack['points'])
                        d = [columns.index(p) for p in points if p in columns]
                        break
                self.diagnosis.append(d)
        else:
            self.diagnosis = [[] for _ in range(len(self.data))]
            
        # 5. Scaling
        if post_scaler is None:
            self.post_scaler = post_scaler_class()
            self.post_scaler.fit(self.data)
        else:
            self.post_scaler = post_scaler

        self.data = self.post_scaler.transform(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < self.window_size:
            pad = self.data[[0]].repeat(self.window_size - idx - 1, axis=0)
            window = np.concatenate((pad, self.data[:idx + 1]), axis=0)
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


class SWaTDataModule:
    def __init__(
        self,
        data_dir: str = "data/",
        window_size: int = 10,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        forecast: bool = False,
    ):
        self.data_dir = os.path.join(data_dir, 'Swat')
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.forecast = forecast

        self.columns = None
        self.scaler = None

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self):
        # --- Load & preprocess normal (train+val) ---
        normal_path = os.path.join(self.data_dir, 'SWaT_Dataset_Normal_v1.csv')
        normal_df = pd.read_csv(normal_path, sep=',', low_memory=False)
        self.columns = [col.strip() for col in normal_df.columns if col not in ['Timestamp', 'Normal/Attack']]
        
        for col in self.columns:
            normal_df[col] = normal_df[col].apply(lambda x: str(x).replace("," , ".")).astype(float)

        # 1. 먼저 시계열 순서대로 train/val 분할
        total_len = len(normal_df)
        train_len = int(total_len * 0.8)
        train_df = normal_df.iloc[:train_len].copy()
        val_df = normal_df.iloc[train_len:].copy()

        # 2. train 데이터에만 fit
        self.scaler = StandardScaler()
        self.scaler.fit(train_df[self.columns].values)

        self.data_train = SWaTDataset(train_df, attacks=[], columns=self.columns,
                                    post_scaler=self.scaler, window_size=self.window_size, forecast=self.forecast)

        self.data_val = SWaTDataset(val_df, attacks=[], columns=self.columns,
                                    post_scaler=self.scaler, window_size=self.window_size, forecast=self.forecast)

        # --- Load & preprocess attack (test) ---
        attack_path = os.path.join(self.data_dir, 'SWaT_Dataset_Attack_v0.csv')
        attack_df = pd.read_csv(attack_path, sep=';', low_memory=False)
        
        for col in self.columns:
            attack_df[col] = attack_df[col].apply(lambda x: str(x).replace("," , ".")).astype(float)
            
        self.data_test = SWaTDataset(attack_df, attacks=swat.attacks, columns=self.columns,
                                    post_scaler=self.scaler, window_size=self.window_size, forecast=self.forecast)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True,
                        num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False,
                        num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.window_size, shuffle=False,
                        num_workers=self.num_workers, pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return self.test_dataloader()
