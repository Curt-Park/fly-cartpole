import json

from fly_cartpole.cli import main, parse_seeds


def test_parse_seeds_accepts_ranges_and_lists():
    assert parse_seeds("0-2,5") == [0, 1, 2, 5]


def test_train_writes_one_file_per_seed(circuit_path, tmp_path):
    main(["train", "--condition", "fly", "--seeds", "0-1", "--episodes", "2", "--workers", "1",
          "--circuit", str(circuit_path), "--results", str(tmp_path)])
    stored = json.loads((tmp_path / "fly" / "seed_1.json").read_text())
    assert stored["condition"] == "fly" and len(stored["lengths"]) == 2


def test_extract_flight_defaults_to_the_cache_and_data_folders():
    from fly_cartpole.cli import build_parser
    from fly_cartpole.paths import CACHE_DIR, FLIGHT_PATH

    args = build_parser().parse_args(["extract-flight"])
    assert (args.command, args.cache, args.output) == ("extract-flight", CACHE_DIR, FLIGHT_PATH)
