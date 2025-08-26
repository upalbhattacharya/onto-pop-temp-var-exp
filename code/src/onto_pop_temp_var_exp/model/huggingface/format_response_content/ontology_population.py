#!/usr/bin/env python

import argparse
import glob
import json
import os
import re

import polars as pl


def llama3_format_response(value: str):

    values = re.sub(
        r"(<\|begin_of_text\|>\s*|<\|eot_id\|>)", "", value
    )  # For older response format
    offset = values.find("<|start_header_id|>assistant<|end_header_id|>")
    if offset != -1:
        values = values[offset + len("<|start_header_id|>assistant<|end_header_id|>") :]
    values = re.sub(
        r"^.+?:", "", values, re.DOTALL
    )  # Removes non-essential starting text
    values = re.sub(r"\b\d+\b", "", values)  # Remove numbers
    values = re.sub(
        r"('|,|\[|\]|\.|\*)", "", values
    )  # Remove other special demarcation characters
    values = re.sub(
        r"\s+", " ", values
    )  # Replace all extra blank spaces/newlines with single spaces
    values = list(filter(None, values.split()))
    return values


parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--response_file",
    help="Response file",
    type=str,
    required=True,
)
parser.add_argument(
    "-l",
    "--label_mapping",
    type=str,
    help="Label mapping file to align responses based on Custom ID",
    required=True,
)

args = parser.parse_args()

response_file_dir = os.path.dirname(args.response_file)

output_dir = response_file_dir

df = pl.read_ndjson(args.response_file)
y_true_df = pl.read_ndjson(args.label_mapping)

join_df = df.join(y_true_df, on="Custom ID")
columns = ["Individual", "Response"]
join_df = join_df.select(columns)
join_df = join_df.with_columns(
    pl.col("Response")
    .map_elements(
        function=lambda x: llama3_format_response(x), return_dtype=pl.List(pl.String)
    )
    .alias("Prediction")
)
print(join_df)

join_df.write_ndjson(os.path.join(output_dir, "predictions.json"))
