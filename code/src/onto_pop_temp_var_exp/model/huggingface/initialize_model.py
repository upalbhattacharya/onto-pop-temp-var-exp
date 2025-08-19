#!/usr/bin/env python

import os

import torch
import torch.nn as nn
from dotenv import load_dotenv
from onto_pop_temp_var_exp.model.huggingface.run_args import RunArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
)

load_dotenv()


def initialize_model(run_args: RunArguments):

    # Get Configuration for device_map
    if run_args.load_in_8bit or run_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=run_args.load_in_8bit,
            load_in_4bit=run_args.load_in_4bit,
        )
        device_map = {"": run_args.device}
        # device_map = "auto"
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        run_args.llm_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=run_args.trust_remote_code,
        torch_dtype=torch_dtype,
        token=os.environ.get("HF_TOKEN"),
        cache_dir="/scratch/bhatt06/.cache",
    )
    return model
