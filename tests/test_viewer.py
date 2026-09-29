import shutil
import subprocess
from pathlib import Path

import pytest

TESTS = Path(__file__).parent / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_viewer_playback_logic():
    result = subprocess.run(["node", "--test", *sorted(str(path) for path in TESTS.glob("*.test.mjs"))],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_browser_fly_and_cartpole_match_python(circuit_path, tmp_path):
    from fly_cartpole.export import export
    from fly_cartpole.params import Hyperparameters

    export(seed=0, episodes=3, params=Hyperparameters(kc_sparsity=0.1), circuit_path=circuit_path,
           left_circuit_path=circuit_path, web_data_dir=tmp_path)
    result = subprocess.run(["node", str(TESTS / "parity.mjs"), str(tmp_path / "model.json")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
