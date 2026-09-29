"""Command line entry point: fly-cartpole <command>."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .paths import CACHE_DIR, CIRCUIT_PATH, RESULTS_DIR, WEB_DATA_DIR

EVALUATION_SEEDS = "0-9"


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        start, _, end = part.partition("-")
        seeds.extend(range(int(start), int(end or start) + 1))
    return seeds


def add_run_options(command: argparse.ArgumentParser, episodes: int) -> None:
    command.add_argument("--episodes", type=int, default=episodes)
    command.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    command.add_argument("--circuit", type=Path, default=CIRCUIT_PATH)
    command.add_argument("--results", type=Path, default=RESULTS_DIR)


def build_parser() -> argparse.ArgumentParser:
    from .run import CONDITIONS

    parser = argparse.ArgumentParser(prog="fly-cartpole", description="A connectome mushroom body learning CartPole.")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="download MaleCNS v1.0 and build data/mb_right.npz")
    extract.add_argument("--cache", type=Path, default=CACHE_DIR)
    extract.add_argument("--circuit", type=Path, default=CIRCUIT_PATH)

    tune = commands.add_parser("tune", help="hyperparameter search on tuning seeds 100-104")
    tune.add_argument("--configs", type=int, default=20)
    add_run_options(tune, episodes=300)

    train = commands.add_parser("train", help="run one condition")
    train.add_argument("--condition", choices=CONDITIONS, required=True)
    train.add_argument("--seeds", default=EVALUATION_SEEDS)
    add_run_options(train, episodes=1000)

    compare = commands.add_parser("compare", help="every condition on evaluation seeds, plot and summary")
    compare.add_argument("--seeds", default=EVALUATION_SEEDS)
    add_run_options(compare, episodes=1000)

    record = commands.add_parser("record", help="record a naive and a trained episode for the web viewer")
    record.add_argument("--seed", type=int, default=0)
    record.add_argument("--web-data", type=Path, default=WEB_DATA_DIR)
    add_run_options(record, episodes=1000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        from .extract import extract

        print(json.dumps(extract(args.cache, args.circuit), indent=2))
        return

    from .params import HYPERPARAMETERS_FILE, load_hyperparameters

    if args.command == "tune":
        from .tune import tune

        print(tune(args.configs, args.episodes, args.workers, args.circuit, args.results))
        return

    params = load_hyperparameters(args.results / HYPERPARAMETERS_FILE)
    if args.command == "train":
        from .run import run_many, save_lengths

        seeds = parse_seeds(args.seeds)
        outputs = run_many([(args.condition, seed, params) for seed in seeds], args.episodes, args.workers, args.circuit)
        for seed, lengths in zip(seeds, outputs):
            path = save_lengths(args.condition, seed, lengths, params, args.results)
            print(f"{path}: final-100 mean {np.mean(lengths[-100:]):.1f}")
    elif args.command == "compare":
        from .report import compare

        print(compare(parse_seeds(args.seeds), args.episodes, params, args.workers, args.circuit, args.results))
    elif args.command == "record":
        from .record import record

        print(json.dumps(record(args.seed, args.episodes, params, args.circuit, args.web_data), indent=2))
