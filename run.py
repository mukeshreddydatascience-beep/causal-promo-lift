#!/usr/bin/env python3
"""Single entry point: python run.py [--config config.yaml]"""

import argparse
import logging
import sys

from src.pipeline import load_config, run


def main():
    parser = argparse.ArgumentParser(
        description="Measure the true lift of a retail promotion.")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to the YAML config file.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    run(load_config(args.config))


if __name__ == "__main__":
    main()
