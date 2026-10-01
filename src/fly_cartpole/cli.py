"""Command line entry point: fly-cartpole <command>."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .paths import (CACHE_DIR, CIRCUIT_PATH, DATA_DIR, FLIGHT_PATH, REFLEX_RESULTS_DIR, RESULTS_DIR, STATION_RESULTS_DIR,
                    WEB_DATA_DIR)

EVALUATION_SEEDS = "70-89"
# Tuning seeds kept improving up to about 3,000 episodes and slipped by 4,000.
TRAINING_EPISODES = 3000


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

    extract = commands.add_parser("extract", help="download MaleCNS v1.0 and build data/mb_right.npz and data/mb_left.npz")
    extract.add_argument("--cache", type=Path, default=CACHE_DIR)
    extract.add_argument("--data", type=Path, default=DATA_DIR)

    extract_flight = commands.add_parser("extract-flight", help="build data/flight.npz: halteres, ocelli and HS cells to the wing steering motors")
    extract_flight.add_argument("--cache", type=Path, default=CACHE_DIR)
    extract_flight.add_argument("--output", type=Path, default=FLIGHT_PATH)

    for name, seeds, description in (("reflex-tune", "100-109", "choose the reflex fly's self-tuning rates on tuning seeds"),
                                     ("reflex-compare", "160-179", "every reflex condition on evaluation seeds, plot and summary")):
        reflex = commands.add_parser(name, help=description)
        reflex.add_argument("--seeds", default=seeds)
        reflex.add_argument("--workers", type=int, default=os.cpu_count() or 1)
        reflex.add_argument("--flight", type=Path, default=FLIGHT_PATH)
        reflex.add_argument("--results", type=Path, default=REFLEX_RESULTS_DIR)

    station = commands.add_parser("station-compare", help="the self-tuned reflex fly with and without a landmark, on fresh seeds")
    station.add_argument("--seeds", default="180-199")
    station.add_argument("--conditions", default="fly-reflex-station-fixed,fly-reflex-adaptive,fly-reflex-station")
    station.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    station.add_argument("--flight", type=Path, default=FLIGHT_PATH)
    # The self-tuning rates come from reflex-tune; nothing is re-tuned for the landmark.
    station.add_argument("--reflex-results", type=Path, default=REFLEX_RESULTS_DIR)
    station.add_argument("--results", type=Path, default=STATION_RESULTS_DIR)

    tune = commands.add_parser("tune", help="hyperparameter search on tuning seeds 100-139")
    tune.add_argument("--configs", type=int, default=40)
    tune.add_argument("--bilateral-episodes", type=int, default=TRAINING_EPISODES)
    add_run_options(tune, episodes=300)

    train = commands.add_parser("train", help="run one condition")
    train.add_argument("--condition", choices=CONDITIONS, required=True)
    train.add_argument("--seeds", default=EVALUATION_SEEDS)
    add_run_options(train, episodes=TRAINING_EPISODES)

    compare = commands.add_parser("compare", help="every condition on evaluation seeds, plot and summary")
    compare.add_argument("--seeds", default=EVALUATION_SEEDS)
    add_run_options(compare, episodes=TRAINING_EPISODES)

    export_flight = commands.add_parser("export-flight", help="self-tune the reflex fly and export it for the live web viewer")
    export_flight.add_argument("--seed", type=int, default=0)
    export_flight.add_argument("--flight", type=Path, default=FLIGHT_PATH)
    export_flight.add_argument("--results", type=Path, default=REFLEX_RESULTS_DIR)
    export_flight.add_argument("--web-data", type=Path, default=WEB_DATA_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        from .extract import extract

        print(json.dumps(extract(args.cache, args.data), indent=2))
        return

    if args.command == "extract-flight":
        from .flight import extract_flight

        print(json.dumps(extract_flight(args.cache, args.output), indent=2))
        return

    if args.command == "export-flight":
        from .flight_export import export_flight
        from .reflex_report import PARAMETERS_FILE, load_reflex_parameters

        params = load_reflex_parameters(args.results / PARAMETERS_FILE)
        print(json.dumps(export_flight(args.seed, params, args.flight, args.web_data), indent=2))
        return

    if args.command == "station-compare":
        from .reflex_report import PARAMETERS_FILE, load_reflex_parameters, station_compare

        params = load_reflex_parameters(args.reflex_results / PARAMETERS_FILE)
        print(station_compare(parse_seeds(args.seeds), args.workers, params, args.flight, args.results,
                              conditions=tuple(args.conditions.split(","))))
        return

    if args.command in ("reflex-tune", "reflex-compare"):
        from .reflex_report import PARAMETERS_FILE, load_reflex_parameters, reflex_compare, reflex_tune

        if args.command == "reflex-tune":
            print(reflex_tune(parse_seeds(args.seeds), args.workers, args.flight, args.results))
        else:
            print(reflex_compare(parse_seeds(args.seeds), args.workers, load_reflex_parameters(args.results / PARAMETERS_FILE),
                                 args.flight, args.results))
        return

    from .params import HYPERPARAMETERS_FILE, load_hyperparameters

    if args.command == "tune":
        from .tune import tune

        print(tune(args.configs, args.episodes, args.workers, args.circuit, args.results, bilateral_episodes=args.bilateral_episodes))
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
