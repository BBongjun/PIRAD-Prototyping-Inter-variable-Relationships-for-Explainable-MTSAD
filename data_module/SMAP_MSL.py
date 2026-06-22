# nasa_scalar_dm.py
import os
from typing import Optional, Literal, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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

class ScalarWindowDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,                 # (N, W, C)
        targets: np.ndarray,                 # (N,)
        forecasts: Optional[np.ndarray] = None  # (N, W, C) or None
    ):
        self.windows = windows.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.forecasts = forecasts.astype(np.float32) if forecasts is not None else None
        self.diagnosis = [[] for _ in range(len(self.targets))] 

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        if self.forecasts is not None:
            f = torch.tensor(self.forecasts[idx], dtype=torch.float32)
            return x, y, f
        return x, y

# === SMAP / MSL DataModule (PSM/SMD/SWaT 스타일) ===
class NASADataModule:
    def __init__(
        self,
        dataset: Literal["SMAP", "MSL"],
        data_dir: str,
        window_size: int = 100,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_scaler: bool = True,
        scaler_class = StandardScaler,
        forecast: bool = False,
    ):
        self.dataset = dataset.upper()
        assert self.dataset in ("SMAP", "MSL")
        self.data_dir = data_dir
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_scaler = use_scaler
        self.scaler_class = scaler_class
        self.forecast = forecast

        self.scaler: Optional[StandardScaler] = None
        self.data_train: Optional[ScalarWindowDataset] = None
        self.data_val:   Optional[ScalarWindowDataset] = None
        self.data_test:  Optional[ScalarWindowDataset] = None

    def _load_npy(self, suffix: str) -> np.ndarray:
        path = os.path.join(self.data_dir, f"{self.dataset}_{suffix}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
        return np.load(path).astype(np.float32)

    def setup(self):
        # 1) 로드
        train_raw = self._load_npy("train")      # (T_tr, C)
        test_raw  = self._load_npy("test")       # (T_te, C)
        test_lab  = self._load_npy("test_label").squeeze()  # (T_te,)

        if test_lab.ndim != 1:
            raise ValueError("test_label must be 1-D (T,)")

        # 2) train 내부 8:2 split (시간 순서 유지)
        T = len(train_raw)
        split = int(T * 0.8)
        tr = train_raw[:split]
        va = train_raw[split:]

        # 비지도 가정: train/val 라벨=0
        tr_lab = np.zeros((len(tr),), dtype=np.float32)
        va_lab = np.zeros((len(va),), dtype=np.float32)

        # 3) scaler: train(앞 80%)에만 fit
        if self.use_scaler:
            self.scaler = self.scaler_class()
            self.scaler.fit(tr)
            tr = self.scaler.transform(tr)
            va = self.scaler.transform(va)
            te = self.scaler.transform(test_raw)
        else:
            self.scaler = None
            te = test_raw

        # 4) 윈도 생성 (끝시점 라벨)
        tr_x, tr_y, tr_f = generate_windows(tr, tr_lab, self.window_size, forecast=self.forecast)
        va_x, va_y, va_f = generate_windows(va, va_lab, self.window_size, forecast=self.forecast)
        te_x, te_y, te_f = generate_windows(te, test_lab, self.window_size, forecast=self.forecast)

        # 5) Dataset 구성
        self.data_train = ScalarWindowDataset(tr_x, tr_y, tr_f)
        self.data_val   = ScalarWindowDataset(va_x, va_y, va_f)
        self.data_test  = ScalarWindowDataset(te_x, te_y, te_f)

        print(f"[{self.dataset}] train_raw={train_raw.shape}, test_raw={test_raw.shape}, "
              f"win={self.window_size}, forecast={self.forecast}")
        print(f"  -> windows: train={len(self.data_train)}, val={len(self.data_val)}, test={len(self.data_test)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
