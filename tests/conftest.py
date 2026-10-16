import os
from pathlib import Path

import pytest

from hrsmpipe.config import load_config


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("HRSM_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture
def cfg(data_root):
    return load_config(None, {"run.name": "t", "run.start": "2024-01-01", "run.end": "2024-01-03"})


def real_data_dir() -> Path | None:
    p = os.environ.get("HRSM_DATA_DIR")
    return Path(p) if p else None
