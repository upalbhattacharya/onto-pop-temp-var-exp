#!/usr/bin/env python

import argparse
import glob
import json
import os
import re

import polars as pl
from onto_pop_temp_var_exp.model.openai.run_args import RunArguments


def gpt_4o_format_response(value: str):
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
    "--response_file_dir",
    help="Response objects directory",
    type=str,
    required=True,
)
parser.add_argument(
    "-r",
    "--run_args",
    help="Run arguments",
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
with open(args.run_args, "r") as f:
    args_raw = f.read()
    run_args = RunArguments.parse_raw(args_raw)

output_dir = args.response_file_dir
path_glob = f"{args.response_file_dir}/batch_output_*.jsonl"
batch_output_files = glob.glob(path_glob)
results = []

# Create Prediction DataFrame
for batch_output_path in batch_output_files:
    with open(batch_output_path, "r") as f:
        for line in f:
            json_object = json.loads(line.strip())
            results.append(
                (
                    json_object["custom_id"],
                    json_object["response"]["body"]["choices"][0]["message"]["content"],
                )
            )

df = pl.DataFrame(results, schema=[("Custom ID", str), ("Response", str)])

y_true_df = pl.read_ndjson(args.label_mapping)
print(results)
print(y_true_df)

join_df = df.join(y_true_df, on="Custom ID")
print(join_df)
columns = ["Individual", "Response"]
join_df = join_df.select(columns)
join_df = join_df.with_columns(
    pl.col("Response")
    .map_elements(
        function=lambda x: gpt_4o_format_response(x), return_dtype=pl.List(pl.String)
    )
    .alias("Prediction")
)
print(join_df)

join_df.write_ndjson(os.path.join(output_dir, "predictions.json"))
