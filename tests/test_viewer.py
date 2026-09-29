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
