import argparse
from procodex.config import *


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-experiment",
        "--experiment",
        help="Experiment No",
        default=100,
        required=False,
        type=int,
    )
    parser.add_argument(
        "-log_dir",
        "--log_dir",
        help="Directory to store logs",
        required=False,
        default=f"{BASE_DIR}/logs/",
        type=str,
    )
    args = vars(parser.parse_args())

    return args
