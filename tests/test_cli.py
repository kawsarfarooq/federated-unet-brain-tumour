"""Device selection should work without requiring multiple GPUs or GPU hardware in tests."""
import sys
from types import SimpleNamespace

import pytest

from brainseg_fl.cli import main


@pytest.mark.parametrize("args, expected", [
    (["--device", "cpu"], "-1"),
    (["--device", "gpu"], "0"),
    (["--device", "gpu", "--gpu", "GPU-test-uuid"], "GPU-test-uuid"),
])
def test_device_selection(monkeypatch, tmp_path, args, expected):
    import os
    observed = []
    # Record changes even when the variable was initially absent, then test the default.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setitem(sys.modules, "brainseg_fl.training", SimpleNamespace(execute=lambda opts: observed.append(opts)))
    common = ["--dataset", str(tmp_path), "--cache", str(tmp_path/"cache"), "--run-dir", str(tmp_path/"run")]
    assert main(common + args) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == expected
    assert len(observed) == 1


def test_invalid_device_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(SystemExit) as error:
        main(["--dataset", str(tmp_path), "--cache", str(tmp_path), "--run-dir", str(tmp_path), "--device", "gpu", "--gpu", "0,1"])
    assert error.value.code == 2
