import os

# Keep statsmodels' BLAS from spawning a thread per core in each test worker.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def data_dirs(tmp_path, monkeypatch):
    """Point every pipeline path at a throwaway directory."""
    monkeypatch.setenv("RA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RA_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path


def monthly(values, start="2022-01-01"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.date_range(start, periods=len(values), freq="MS"))


@pytest.fixture
def growing_series():
    """Four years of a random walk with 2% monthly drift."""
    rng = np.random.default_rng(0)
    return monthly(100 * np.exp(np.cumsum(0.02 + rng.normal(0, 0.01, 48))))
