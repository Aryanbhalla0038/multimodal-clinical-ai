"""Vitals preprocessing — resample to hourly, forward/back-fill, scale, pad/truncate."""
from __future__ import annotations
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

VITALS = ["heart_rate", "sbp", "dbp", "spo2", "resp_rate", "temp"]
SEQ_LEN = 48  # 48 hours of hourly samples


def preprocess_vitals(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    vitals: Iterable[str] = VITALS,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[np.ndarray, StandardScaler]:
    """Resample to hourly, ffill/bfill, scale, pad/truncate to seq_len."""
    vitals = list(vitals)
    if "charttime" in df.columns:
        df = df.set_index("charttime")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df[vitals].resample("1h").mean()
    df = df.ffill().bfill()
    df = df.iloc[:seq_len]

    if len(df) < seq_len:
        n_pad = seq_len - len(df)
        last_t = df.index[-1] if len(df) else pd.Timestamp("2000-01-01")
        pad_idx = pd.date_range(last_t, periods=n_pad + 1, freq="1h")[1:]
        pad = pd.DataFrame(0.0, index=pad_idx, columns=vitals)
        df = pd.concat([df, pad])

    df = df.fillna(0.0)
    arr = df.values.astype(np.float32)

    if scaler is None and fit_scaler:
        scaler = StandardScaler()
        arr = scaler.fit_transform(arr)
    elif scaler is not None:
        arr = scaler.transform(arr)

    return arr.astype(np.float32), scaler  # type: ignore[return-value]


def synthetic_vitals(seed: int | None = None, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Plausible synthetic vitals (zero-mean, unit-std), shape (seq_len, 6)."""
    rng = np.random.default_rng(seed)
    base = np.array([80.0, 120.0, 75.0, 97.0, 16.0, 36.8], dtype=np.float32)
    noise = np.array([8.0, 12.0, 8.0, 1.5, 2.5, 0.4], dtype=np.float32)
    t = np.linspace(0, 4 * np.pi, seq_len, dtype=np.float32)[:, None]
    drift = np.sin(t) * noise * 0.5
    out = base + drift + rng.normal(0, noise, size=(seq_len, 6)).astype(np.float32)
    out = (out - out.mean(axis=0)) / (out.std(axis=0) + 1e-6)
    return out.astype(np.float32)
