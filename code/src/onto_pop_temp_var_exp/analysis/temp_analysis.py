#!/usr/bin/env python

"""Temperature Variation Analysis"""

from collections import defaultdict
from typing import Any

import polars as pl


def hypothesis_1(preds: dict[list[pl.DataFrame]]) -> None:
    """Compute and plot the average number of directly-asserted
    concepts predicted for each query"""
    # TODO: Improve docstring

    responses = defaultdict(lambda: defaultdict(list))
    for temp, run_preds in preds.items():
        # Get first concept simultaneously from all run predictions
        for i in range(run_preds[0].select(pl.len()).item()):
            assertion = [
                run_pred.row(i, named=True)["Prediction"][0] for run_pred in run_preds
            ]
            responses[temp][run_preds[0].row(i, named=True)["Individual"]] = assertion
    print(responses)


if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--runs_dirs",
        nargs="+",
        type=str,
        required=True,
        help="Runs directories for predictions",
    )
    parser.add_argument(
        "-t",
        "--temperatures",
        nargs="+",
        type=float,
        required=True,
        help="Temperatures for run directories",
        )
    args = parser.parse_args()

    # predictions = 
    for r_dir in args.runs_dirs:
        for run in os.listdir(r_dir):
            
            with open(os.path.join(r_dir, run, "predictions.json"), 
