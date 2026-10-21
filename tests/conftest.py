import os

# Keep statsmodels' BLAS from spawning a thread per core in each test worker.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def data_dirs(tmp_path, monkeypatch):
    """Point every pipeline path at a throwaway directory."""
    monkeypatch.setenv("RA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RA_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path


@pytest.fixture
def trending_series():
    rng = np.random.default_rng(0)
    return list(np.round(50 + 0.5 * np.arange(40) + rng.normal(0, 0.3, 40), 2))
