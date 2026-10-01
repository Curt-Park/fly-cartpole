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


def test_reflex_commands_default_to_their_own_seeds_and_results():
    from fly_cartpole.cli import build_parser
    from fly_cartpole.paths import REFLEX_RESULTS_DIR

    compare = build_parser().parse_args(["reflex-compare"])
    tune = build_parser().parse_args(["reflex-tune"])
    assert (compare.seeds, compare.results, tune.seeds, tune.results) == ("160-179", REFLEX_RESULTS_DIR, "100-109", REFLEX_RESULTS_DIR)


def test_station_compare_defaults_to_fresh_seeds_and_its_own_results():
    from fly_cartpole.cli import build_parser
    from fly_cartpole.paths import STATION_RESULTS_DIR

    args = build_parser().parse_args(["station-compare"])
    assert (args.seeds, args.results) == ("180-199", STATION_RESULTS_DIR)


def test_the_viewer_is_exported_from_the_reflex_fly():
    import pytest

    from fly_cartpole.cli import build_parser
    from fly_cartpole.paths import FLIGHT_PATH, REFLEX_RESULTS_DIR, WEB_DATA_DIR

    args = build_parser().parse_args(["export-flight"])
    assert (args.seed, args.flight, args.results, args.web_data) == (0, FLIGHT_PATH, REFLEX_RESULTS_DIR, WEB_DATA_DIR)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["export"])
