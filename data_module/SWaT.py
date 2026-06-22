# data_module/swat_data_module.py

# import os
# import pandas as pd
# import numpy as np
# import torch

# from datetime import datetime
# from typing import Any, Optional, Tuple
# from torch.utils.data import Dataset, DataLoader, Subset
# from sklearn.preprocessing import StandardScaler
# from data_module.components import swat

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
    
# class SWaTDataset(Dataset):
#     def __init__(
#         self,
#         df: pd.DataFrame,
#         attacks: Optional[list[dict]],
#         columns: list[str],
#         window_size: int,
#         forecast: bool = False,
#         post_scaler_class: Any = StandardScaler,
#         post_scaler: Optional[Any] = None,
#     ):
#         super().__init__()
#         self.window_size = window_size
#         self.forecast = forecast
#         self.columns = columns

#         # Timestamp, Label 처리
#         self.timestamps = pd.to_datetime(df['Timestamp'].str.strip(), format="%d/%m/%Y %I:%M:%S %p")
#         if 'Normal/Attack' in df.columns:
#             labels = df['Normal/Attack'].astype(str).apply(lambda x: float(x.strip().lower() != 'normal')).values
#             df = df.drop(columns=['Normal/Attack'])
#         else:
#             labels = np.zeros(len(df), dtype=np.float32)

#         # Feature
#         self.data = df[columns].values

#         # Scaling
#         if post_scaler is None:
#             self.post_scaler = post_scaler_class()
#             self.post_scaler.fit(self.data)
#         else:
#             self.post_scaler = post_scaler
#         self.data = self.post_scaler.transform(self.data)

#         # diagnosis 생성
#         self.diagnosis = [[] for _ in range(len(self.data))]
#         if attacks:
#             for i, t in enumerate(self.timestamps):
#                 for attack in attacks:
#                     if attack['start_time_dt'] <= t <= attack['end_time_dt']:
#                         labels[i] = 1.0
#                         points = map(lambda p: p.replace('-', ''), attack['points'])
#                         self.diagnosis[i] = [columns.index(p) for p in points if p in columns]
#                         break

#         # ✅ 윈도우 사전 생성
#         self.windows, self.targets, self.forecasts = generate_windows(
#             self.data, labels, window_size, forecast
#         )

#     def __len__(self):
#         return len(self.windows)

#     def __getitem__(self, idx):
#         x = torch.tensor(self.windows[idx], dtype=torch.float32)
#         y = torch.tensor(self.targets[idx], dtype=torch.float32)
#         if self.forecast:
#             f = torch.tensor(self.forecasts[idx], dtype=torch.float32)
#             return x, y, f
#         return x, y
    
# class SWaTDataModule:
#     def __init__(
#         self,
#         data_dir: str = "data/",
#         window_size: int = 10,
#         batch_size: int = 64,
#         num_workers: int = 0,
#         pin_memory: bool = False,
#         forecast: bool = False,
#     ):
#         self.data_dir = os.path.join(data_dir, 'Swat')
#         self.window_size = window_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.forecast = forecast

#         self.columns = None
#         self.scaler = None
#         self.data_train = None
#         self.data_val = None
#         self.data_test = None

#     def setup(self):
#         import pandas as pd
#         from data_module.components import swat

#         # Load normal data
#         normal_path = os.path.join(self.data_dir, 'SWaT_Dataset_Normal_v1.csv')
#         normal_df = pd.read_csv(normal_path, sep=',', low_memory=False)
#         self.columns = [col.strip() for col in normal_df.columns if col not in ['Timestamp', 'Normal/Attack']]

#         for col in self.columns:
#             normal_df[col] = normal_df[col].apply(lambda x: str(x).replace("," , ".")).astype(float)

#         train_len = int(len(normal_df) * 0.8)
#         train_df = normal_df.iloc[:train_len].copy()
#         val_df = normal_df.iloc[train_len:].copy()

#         self.scaler = StandardScaler()
#         self.scaler.fit(train_df[self.columns].values)

#         self.data_train = SWaTDataset(train_df, attacks=[], columns=self.columns,
#                                       post_scaler=self.scaler, window_size=self.window_size, forecast=self.forecast)
#         self.data_val = SWaTDataset(val_df, attacks=[], columns=self.columns,
#                                     post_scaler=self.scaler, window_size=self.window_size, forecast=self.forecast)

#         # Load attack data
#         attack_path = os.path.join(self.data_dir, 'SWaT_Dataset_Attack_v0.csv')
#         attack_df = pd.read_csv(attack_path, sep=';', low_memory=False)
#         for col in self.columns:
#             attack_df[col] = attack_df[col].apply(lambda x: str(x).replace("," , ".")).astype(float)

#         self.data_test = SWaTDataset(attack_df, attacks=swat.attacks, columns=self.columns,
#                                      post_scaler=self.scaler, window_size=self.window_size, forecast=self.forecast)

#     def train_dataloader(self):
#         return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True,
#                           num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def val_dataloader(self):
#         return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False,
#                           num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def test_dataloader(self):
#         return DataLoader(self.data_test, batch_size=self.window_size, shuffle=False,
#                           num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def predict_dataloader(self):
#         return self.test_dataloader()


#### 위에는 미리 winodw 만드는거라 메모리 많이 잡아먹음 ###

# import os
# import pandas as pd
# import numpy as np
# import torch

# from datetime import datetime
# from typing import Any, Optional, Tuple
# from torch.utils.data import Dataset, DataLoader, Subset
# from sklearn.preprocessing import StandardScaler
# from data_module.components import swat


# class SWaTDataset(Dataset):
#     def __init__(
#         self,
#         df: pd.DataFrame,
#         attacks: Optional[list[dict]],
#         columns: list[str],
#         window_size: int,
#         forecast: bool = False,
#         post_scaler_class: Any = StandardScaler,
#         post_scaler: Optional[Any] = None,
#     ):
#         super().__init__()
#         self.window_size = window_size
#         self.forecast = forecast
#         self.columns = columns

#         # Timestamp, Labels, Features
#         self.timestamps = pd.to_datetime(df['Timestamp'].str.strip(), format="%d/%m/%Y %I:%M:%S %p")
#         if 'Normal/Attack' in df.columns:
#             labels = df['Normal/Attack'].astype(str).apply(lambda x: float(x.strip().lower() != 'normal')).values
#             df = df.drop(columns=['Normal/Attack'])
#         else:
#             labels = np.zeros(len(df), dtype=np.float32)
            
#         self.data = df[columns].values
#         self.labels = labels

#         # Apply attack labels and diagnosis
#         self.diagnosis = []
#         if attacks:
#             for i, t in enumerate(self.timestamps):
#                 d = []
#                 for attack in attacks:
#                     if attack['start_time_dt'] <= t <= attack['end_time_dt']:
#                         self.labels[i] = 1.0
#                         points = map(lambda p: p.replace('-', ''), attack['points'])
#                         d = [columns.index(p) for p in points if p in columns]
#                         break
#                 self.diagnosis.append(d)
#         else:
#             self.diagnosis = [[] for _ in range(len(self.data))]
            
#         # 5. Scaling
#         if post_scaler is None:
#             self.post_scaler = post_scaler_class()
#             self.post_scaler.fit(self.data)
#         else:
#             self.post_scaler = post_scaler

#         self.data = self.post_scaler.transform(self.data)
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         if idx < self.window_size:
#             pad = self.data[[0]].repeat(self.window_size - idx - 1, axis=0)
#             window = np.concatenate((pad, self.data[:idx + 1]), axis=0)
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
#         window_size: int = 10,
#         batch_size: int = 64,
#         num_workers: int = 0,
#         pin_memory: bool = False,
#         forecast: bool = False,
#     ):
#         self.data_dir = os.path.join(data_dir, 'Swat')
#         self.window_size = window_size
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.forecast = forecast

#         self.columns = None
#         self.scaler = None

#         self.data_train = None
#         self.data_val = None
#         self.data_test = None

#     def setup(self):
#         # --- Load & preprocess normal (train+val) ---
#         normal_path = os.path.join(self.data_dir, 'SWaT_Dataset_Normal_v1.csv')
#         normal_df = pd.read_csv(normal_path, sep=',', low_memory=False)
#         self.columns = [col.strip() for col in normal_df.columns if col not in ['Timestamp', 'Normal/Attack']]
        
#         for col in self.columns:
#             normal_df[col] = normal_df[col].apply(lambda x: str(x).replace("," , ".")).astype(float)

#         # 1. 먼저 시계열 순서대로 train/val 분할
#         total_len = len(normal_df)
#         train_len = int(total_len * 0.8)
#         train_df = normal_df.iloc[:train_len].copy()
#         val_df = normal_df.iloc[train_len:].copy()

#         # 2. train 데이터에만 fit
#         self.scaler = StandardScaler()
#         self.scaler.fit(train_df[self.columns].values)

#         self.data_train = SWaTDataset(train_df, attacks=[], columns=self.columns,
#                                     post_scaler=self.scaler, window_size=self.window_size, forecast=self.forecast)

#         self.data_val = SWaTDataset(val_df, attacks=[], columns=self.columns,
#                                     post_scaler=self.scaler, window_size=self.window_size, forecast=self.forecast)

#         # --- Load & preprocess attack (test) ---
#         attack_path = os.path.join(self.data_dir, 'SWaT_Dataset_Attack_v0.csv')
#         attack_df = pd.read_csv(attack_path, sep=';', low_memory=False)
        
#         for col in self.columns:
#             attack_df[col] = attack_df[col].apply(lambda x: str(x).replace("," , ".")).astype(float)
            
#         self.data_test = SWaTDataset(attack_df, attacks=swat.attacks, columns=self.columns,
#                                     post_scaler=self.scaler, window_size=self.window_size, forecast=self.forecast)

#     def train_dataloader(self):
#         return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True,
#                         num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def val_dataloader(self):
#         return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False,
#                         num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def test_dataloader(self):
#         return DataLoader(self.data_test, batch_size=self.window_size, shuffle=False,
#                         num_workers=self.num_workers, pin_memory=self.pin_memory)

#     def predict_dataloader(self):
#         return self.test_dataloader()


import os
import pandas as pd
import numpy as np
import torch

from datetime import datetime
from typing import Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
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
        use_scaler: bool = True,              # 🔹 StandardScaler 적용 여부
        post_scaler_class: Any = StandardScaler,
        post_scaler: Optional[Any] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.forecast = forecast
        self.columns = columns
        self.use_scaler = use_scaler

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
            
        # 🔹 Scaling 적용 여부 제어
        if self.use_scaler:
            if post_scaler is None:
                self.post_scaler = post_scaler_class()
                self.post_scaler.fit(self.data)
            else:
                self.post_scaler = post_scaler
            self.data = self.post_scaler.transform(self.data)
        else:
            self.post_scaler = None
            self.data = self.data.astype(np.float32)

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
        use_scaler: bool = True,          # 🔹 StandardScaler 적용 여부 플래그
    ):
        self.data_dir = os.path.join(data_dir, 'Swat')
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.forecast = forecast
        self.use_scaler = use_scaler

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

        # 1. train/val split
        total_len = len(normal_df)
        train_len = int(total_len * 0.8)
        train_df = normal_df.iloc[:train_len].copy()
        val_df = normal_df.iloc[train_len:].copy()

        # 2. train 데이터에만 fit (use_scaler=True일 때만)
        if self.use_scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(train_df[self.columns].values)
        else:
            self.scaler = None

        self.data_train = SWaTDataset(train_df, attacks=[], columns=self.columns,
                                    post_scaler=self.scaler, window_size=self.window_size,
                                    forecast=self.forecast, use_scaler=self.use_scaler)

        self.data_val = SWaTDataset(val_df, attacks=[], columns=self.columns,
                                    post_scaler=self.scaler, window_size=self.window_size,
                                    forecast=self.forecast, use_scaler=self.use_scaler)

        # --- Load & preprocess attack (test) ---
        attack_path = os.path.join(self.data_dir, 'SWaT_Dataset_Attack_v0.csv')
        attack_df = pd.read_csv(attack_path, sep=';', low_memory=False)
        
        for col in self.columns:
            attack_df[col] = attack_df[col].apply(lambda x: str(x).replace("," , ".")).astype(float)
            
        self.data_test = SWaTDataset(attack_df, attacks=swat.attacks, columns=self.columns,
                                    post_scaler=self.scaler, window_size=self.window_size,
                                    forecast=self.forecast, use_scaler=self.use_scaler)

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