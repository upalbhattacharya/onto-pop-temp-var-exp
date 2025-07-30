#!/usr/bin/env python

import re
from typing import Optional

from onto_pop_temp_var_exp.definitions import DATA_DIR, RESULTS_DIR, ROOT_DIR
from pydantic import BaseModel, Field


class RunArguments(BaseModel):
    input_file: Optional[str] = Field(
        default=None, metadata={"help": "Dataset file to load"}
    )
    examples_file: Optional[str] = Field(
        default=None, metadata={"help": "Dataset of examples to load for few-shot"}
    )
    output_dir: Optional[str] = Field(
        default=None, metadata={"help": "Directory to save run data"}
    )
    prompt_strategy_type: Optional[str] = Field(
        default=None, metadata={"help": "Prompting strategy"}
    )
    llm_name: Optional[str] = Field(
        default=None,
        metadata={"help": "Name of model to load"},
    )
    max_tokens: Optional[int] = Field(
        default=1, metadata={"help": "Maximum number of tokens to generate"}
    )
    temperature: Optional[float] = Field(
        default=None, metadata={"help": "Temperature for generation"}
    )
    system_message: Optional[str] = Field(
        default=None, metadata={"help": "System message to use for the model"}
    )
    user_prompt_template: Optional[str] = Field(
        default=None,
        metadata={"help": "User input template to use for text inputs to model"},
    )
    regex: Optional[list[str | None]] = Field(
        default=None,
        min_length=2,
        max_length=2,
        metadata={"help": "Regex for text replacement"},
    )
    description: Optional[str] = Field(
        default=None,
        metadata={"help": "Description of the task"},
    )
    kwargs: Optional[dict] = Field(
        default={},
        metadata={
            "help": "Named extra arguments. (Used for different types of prompts)"
        },
    )

    def model_post_init(self, __context):

        self.output_dir = "output" if self.output_dir is None else self.output_dir
        self.input = re.sub(r"{DATA_DIR}", DATA_DIR)
        self.examples_file = re.sub(r"{DATA_DIR}", DATA_DIR)


if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--run_args_file", type=str, required=True, help="`run_args` file to load"
    )
    args = parser.parse_args()

    with open(
        args.run_args_file,
        "r",
    ) as f:
        raw = f.read()
        print(raw)
        run_args = RunArguments.parse_raw(raw)
    print(run_args)
    print(run_args.dict())
    with open("test.json", "w") as f:
        model_dump = run_args.model_dump()
        json.dump(model_dump, f, indent=4)
