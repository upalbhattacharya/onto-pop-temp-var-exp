#!/usr/bin/env python

import logging
import math
import os

import polars as pl
from onto_pop_temp_var_exp.model.openai.dataset.ontology_population import (
    OntologyPopulationRankedRetrievalDataset,
)
from onto_pop_temp_var_exp.model.openai.run_args import RunArguments
from tqdm import tqdm


def create_ranked_retrieval_batch(
    test_data, run_args, **kwargs
) -> (pl.DataFrame, dict):
    tasks = []
    label_mapping = []
    num_samples = len(test_data)
    test_data = iter(test_data)

    for i in tqdm(range(num_samples)):
        print(i)
        print(next(test_data))
        inst, messages, label = next(test_data)
        if run_args.llm_name == "o1-preview":
            task = {
                "custom_id": f"task-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": run_args.llm_name,
                    "messages": messages,
                    "max_completion_tokens": run_args.max_tokens,
                    "temperature": run_args.temperature,
                },
            }
        else:
            task = {
                "custom_id": f"task-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": run_args.llm_name,
                    "messages": messages,
                    "max_tokens": run_args.max_tokens,
                    "temperature": run_args.temperature,
                },
            }
        tasks.append(task)
        label_mapping.append((f"task-{i}", inst, label))

        if kwargs.get("stop", None) is not None and i == kwargs["stop"]:
            break

    df = pl.DataFrame(
        label_mapping,
        schema=[
            ("Custom ID", str),
            ("Individual", str),
            ("Member", list[str]),
        ],
    )

    return tasks, df


if __name__ == "__main__":
    import argparse
    import json
    import logging.config

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--args_file",
        help="Path to `RunArguments` file",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    with open(args.args_file, "r") as f:
        args_raw = f.read()
        run_args = RunArguments.parse_raw(args_raw)

    # Get filename to name output directory
    dir_name = os.path.splitext(os.path.basename(args.args_file))[0]
    output_dir = os.path.join(run_args.output_dir, dir_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "params.json"), "w") as f:
        params_dump = run_args.model_dump()
        json.dump(params_dump, f, indent=4)

    test_data = OntologyPopulationRankedRetrievalDataset(
        in_file=run_args.input_file,
        system_message=run_args.system_message,
        user_prompt_template=run_args.user_prompt_template,
        regex=run_args.regex,
        examples_file=run_args.examples_file,
        llm_name=run_args.llm_name,
        **run_args.kwargs,
    )
    tasks, df = create_ranked_retrieval_batch(test_data, run_args)
    df.write_ndjson(os.path.join(output_dir, "label_mapping.json"))
    iterator = iter(tasks)

    for i in range(math.ceil(len(tasks) / 50000)):
        with open(
            os.path.join(output_dir, f"batch_tasks_{i + 1}.jsonl"),
            "w",
        ) as f:
            try:
                for j in range(50000):
                    f.write(json.dumps(next(iterator)) + "\n")
            except StopIteration:
                exit(0)
