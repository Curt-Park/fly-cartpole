"""Repository locations shared by every command."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache" / "malecns"
RESULTS_DIR = ROOT / "results"
WEB_DATA_DIR = ROOT / "web" / "data"
CIRCUIT_PATH = DATA_DIR / "mb_right.npz"
LEFT_CIRCUIT_PATH = DATA_DIR / "mb_left.npz"
FLIGHT_PATH = DATA_DIR / "flight.npz"
