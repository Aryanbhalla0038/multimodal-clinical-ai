"""Unit tests for data preprocessing — run with `pytest tests/`."""
import numpy as np
import pandas as pd
import pytest

from src.data.vitals_preprocessor import VITALS, SEQ_LEN, preprocess_vitals, synthetic_vitals
from src.utils.labels import LABEL_NAMES, NUM_LABELS, apply_uncertainty_policy


def _dummy_vitals_df(n_rows: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="1h")
    data = {c: np.random.rand(n_rows) for c in VITALS}
    return pd.DataFrame(data, index=idx).rename_axis("charttime").reset_index()


def test_vitals_output_shape_short_input():
    df = _dummy_vitals_df(30)
    arr, _ = preprocess_vitals(df)
    assert arr.shape == (SEQ_LEN, 6), f"got {arr.shape}"


def test_vitals_output_shape_long_input():
    df = _dummy_vitals_df(72)
    arr, _ = preprocess_vitals(df)
    assert arr.shape == (SEQ_LEN, 6)


def test_vitals_no_nans():
    df = _dummy_vitals_df(10)
    arr, _ = preprocess_vitals(df)
    assert not np.isnan(arr).any()


def test_vitals_dtype_float32():
    df = _dummy_vitals_df(20)
    arr, _ = preprocess_vitals(df)
    assert arr.dtype == np.float32


def test_synthetic_vitals_shape():
    arr = synthetic_vitals(seed=0)
    assert arr.shape == (SEQ_LEN, 6)
    assert arr.dtype == np.float32


def test_label_count():
    assert NUM_LABELS == 14
    assert len(LABEL_NAMES) == 14


def test_uncertainty_u_zeros():
    df = pd.DataFrame({n: [1.0, 0.0, -1.0, np.nan] for n in LABEL_NAMES})
    out = apply_uncertainty_policy(df, policy="u_zeros")
    for n in LABEL_NAMES:
        assert list(out[n]) == [1.0, 0.0, 0.0, 0.0]


def test_uncertainty_u_ones_for_specific_labels():
    df = pd.DataFrame({n: [-1.0] for n in LABEL_NAMES})
    out = apply_uncertainty_policy(df, policy="u_zeros", u_ones_labels=["Edema", "Atelectasis"])
    assert out["Edema"].iloc[0] == 1.0
    assert out["Atelectasis"].iloc[0] == 1.0
    assert out["Pneumonia"].iloc[0] == 0.0


def test_uncertainty_u_ignore_produces_nan():
    df = pd.DataFrame({n: [-1.0] for n in LABEL_NAMES})
    out = apply_uncertainty_policy(df, policy="u_ignore")
    assert np.isnan(out["Pneumonia"].iloc[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
