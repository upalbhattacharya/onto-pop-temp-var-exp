#!/usr/bin/env python

import argparse
import logging
import math
from typing import Dict, Union

import polars as pl
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as prfs


def ranked_retrieval(
    true_df: pl.DataFrame, pred_df: pl.DataFrame, **kwargs
) -> Dict[str, Union[Dict[str, float], float]]:
    datapoints = pl.Series(true_df.select(pl.first()))
    y_true = true_df["Ranked List"].to_list()
    y_pred = pred_df["Prediction"].to_list()
    print(kwargs["k"])
    print(y_true)
    print(y_pred)

    def r_prec_score(y_t, y_p):
        return 1.0 * (sum([pred in y_t for pred in y_p[: len(y_t)]])) / (len(y_t))

    r_prec = {
        k: r_prec_score(y_t, y_p) for k, y_t, y_p in zip(datapoints, y_true, y_pred)
    }

    r_prec = {k: v for k, v in sorted(r_prec.items(), key=lambda x: x[1], reverse=True)}

    avg_r_prec = 1.0 * sum(list(r_prec.values())) / len(list(r_prec.keys()))

    def ap_k_score(y_t, y_p, k):
        if len(y_p) > k:
            y_p = y_p[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(y_p):
            if p in y_t and p not in y_p[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not y_t:
            return 0.0

        return score / min(len(y_t), k)

    ap_k = {
        k: ap_k_score(y_t, y_p, int(kwargs["k"]))
        for k, y_t, y_p in zip(datapoints, y_true, y_pred)
    }

    ap_k = {k: v for k, v in sorted(ap_k.items(), key=lambda x: x[1], reverse=True)}

    map_k = 1.0 * sum(list(ap_k.values())) / len(list(ap_k.keys()))

    ap_1 = {
        k: ap_k_score(y_t, y_p, 1) for k, y_t, y_p in zip(datapoints, y_true, y_pred)
    }

    ap_1 = {k: v for k, v in sorted(ap_1.items(), key=lambda x: x[1], reverse=True)}

    map_1 = 1.0 * sum(list(ap_1.values())) / len(list(ap_1.keys()))

    def dcg_k(y_t, y_p, k):
        return sum(
            [
                (2 ** (1.0 / (i + 1)) - 1) / math.log(i + 2, 2)
                for i, item in enumerate(y_p[: min(len(y_t), k)])
                if item in y_t[:k]
            ]
        )

    ndcg_k = sum(
        [
            1.0 * dcg_k(y_t, y_p, int(kwargs["k"])) / dcg_k(y_t, y_t, int(kwargs["k"]))
            for y_t, y_p in zip(y_true, y_pred)
        ]
    ) / len(y_true)

    return {
        "r_prec": r_prec,
        "macro_r_prec": avg_r_prec,
        "ap_k": ap_k,
        f'map_{kwargs["k"]}': map_k,
        "map_1": map_1,
        f'ndcg_{kwargs["k"]}': ndcg_k,
    }


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


if __name__ == "__main__":
    import argparse
    import json
    import os

    import polars as pl

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-yt", "--y_true", help="Path to true labels", type=str, required=True
    )
    parser.add_argument(
        "-yp", "--y_pred", help="Path to prediction labels", type=str, required=True
    )
    parser.add_argument(
        "-k",
        "--kwargs",
        nargs="*",
        help="Extra Named Arguments",
        action=ParseKwargs,
    )
    args = parser.parse_args()

    if not args.kwargs:
        args.kwargs = {}

    output_dir = os.path.dirname(args.y_pred)
    y_true = pl.read_ndjson(args.y_true)
    y_pred = pl.read_ndjson(args.y_pred)

    metrics = ranked_retrieval(y_true, y_pred, **args.kwargs)
    with open(os.path.join(output_dir, "pred_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
