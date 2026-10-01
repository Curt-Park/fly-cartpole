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
def test_browser_reflex_and_cartpole_match_python(flight_path, tmp_path):
    from fly_cartpole.flight_export import export_flight
    from fly_cartpole.reflex_report import ReflexParameters

    export_flight(seed=0, params=ReflexParameters(adapt_episodes=3), flight_path=flight_path, web_data_dir=tmp_path)
    result = subprocess.run(["node", str(TESTS / "reflex_parity.mjs"), str(tmp_path / "flight.json")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_committed_viewer_data_runs_in_the_browser_as_in_python():
    data = Path(__file__).parents[1] / "web" / "data" / "flight.json"
    result = subprocess.run(["node", str(TESTS / "reflex_parity.mjs"), str(data)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
