#!/usr/bin/env python

from typing import Optional

from pydantic import BaseModel, Field
from transformers import TrainingArguments


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
        default=None, metadata={"help": "Temperature used for generation"}
    )
    load_in_8bit: Optional[bool] = Field(
        default=False, metadata={"help": "Load model with 8-bit precision"}
    )
    load_in_4bit: Optional[bool] = Field(
        default=True, metadata={"help": "Load model with 4-bit precision"}
    )
    device: Optional[int] = Field(default=0, metadata={"help": "GPU to load model on"})
    trust_remote_code: Optional[bool] = Field(
        default=True, metadata={"help": "Enable `trust_remote_code`"}
    )
    training_args: Optional[TrainingArguments] = Field(
        default=None, metadata={"help": "HuggingFace `TrainingArguments` for a Trainer"}
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

        # Validation
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Cannot load in 4 bit and 8 bit simultaneously")

        self.output_dir = re.sub(r"{RESULTS_DIR}", str(RESULTS_DIR), self.output_dir)
        self.input_file = re.sub(r"{DATA_DIR}", str(DATA_DIR), self.input_file)
        self.examples_file = re.sub(r"{DATA_DIR}", str(DATA_DIR), self.examples_file)

        if self.training_args is not None:
            self.training_args.output_dir = re.sub(
                r"{RESULTS_DIR}", str(RESULTS_DIR), self.output_dir
            )
            self.training_args.logging_dir = re.sub(
                r"{RESULTS_DIR}", str(RESULTS_DIR), self.output_dir
            )


if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--run_args_file",
        help="`run_args` file to load for checking of model",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    # Quick Test
    with open(args.run_args_file, "r") as f:
        raw = f.read()
        run_args = RunArguments.parse_raw(raw)
    print(run_args)
    print(run_args.dict())
    with open("test.json", "w") as f:
        # args_save = run_args.to_dict()
        model_dump = run_args.model_dump()
        json.dump(model_dump, f, indent=4)
    print(run_args.dict())
    with open("test.json", "w") as f:
        # args_save = run_args.to_dict()
        model_dump = run_args.model_dump()
        json.dump(model_dump, f, indent=4)
