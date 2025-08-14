#!/usr/bin/env python

"""
Module to generate different datasets for various prompts for Llama3 models.
"""

import re
from typing import Optional

import polars as pl
from torch.utils.data import Dataset


class OntologyPopulationRankedRetrievalDataset(Dataset):
    """Generate Ranked retrieval prompts for class assertions"""

    def __init__(
        self,
        in_file: str,
        system_message: str,
        user_prompt_template: str,
        regex: Optional[list[str]] = None,
        examples_file: Optional[str] = None,
        llm_name: Optional[str] = "gpt-4o",
        **kwargs,
    ):
        self.df = pl.read_ndjson(in_file)
        self.examples = (
            pl.read_ndjson(examples_file) if examples_file is not None else None
        )
        self.system_message: str = system_message
        self.user_prompt_template: str = user_prompt_template
        self.regex = regex
        self.extra_args = kwargs
        self.classes = list(
            set([cls for items in self.df["Ranked List"].to_list() for cls in items])
        )
        if not self.extra_args:
            self.extra_args = {}
        self.llm_name = llm_name

    def __len__(self):
        return self.df.select(pl.len()).item()

    def generate_examples(self):
        example_print = ["---"]
        for i, row in enumerate(self.examples.rows()):
            ind = (
                row[0]
                if not self.regex
                else (
                    re.sub(
                        rf"{self.regex[0]}",
                        self.regex[1] if not self.regex[1] else f"-{i}",
                        row[0],
                    )
                )
            )
            example_print.append(ind)
            example_print.extend([f"{j+1}. {val}" for j, val in enumerate(row[1])])
            example_print.append("\n")
        return "\n".join(example_print)

    def __getitem__(self, idx):
        ind, label = self.df.row(idx)
        ind_formatted = (
            re.sub(
                rf"{self.regex[0]}",
                self.regex[1] if self.regex[1] is not None else f"-{idx}",
                ind,
            )
            if self.regex is not None
            else ind
        )
        if self.examples is not None:
            if self.llm_name == "o1-preview":
                messages = [
                    {
                        "role": "user",
                        "content": self.system_message.format(
                            **self.extra_args,
                            classes=self.classes,
                            examples=self.generate_examples(),
                        )
                        + "\n"
                        + self.user_prompt_template.format(ind_formatted),
                    },
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": self.system_message.format(
                            **self.extra_args,
                            classes=self.classes,
                            examples=self.generate_examples(),
                        ),
                    },
                    {
                        "role": "user",
                        "content": self.user_prompt_template.format(ind_formatted),
                    },
                ]
        else:
            if self.llm_name == "o1-preview":
                messages = [
                    {
                        "role": "user",
                        "content": self.system_message.format(
                            **self.extra_args,
                            classes=self.classes,
                        )
                        + "\n"
                        + self.user_prompt_template.format(ind_formatted),
                    },
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": self.system_message.format(
                            **self.extra_args,
                            classes=self.classes,
                        ),
                    },
                    {
                        "role": "user",
                        "content": self.user_prompt_template.format(ind_formatted),
                    },
                ]
        print(ind, messages, label)
        return (
            ind,
            messages,
            label,
        )


if __name__ == "__main__":
    import argparse

    from CASE_contradictory_mAP.model.open_ai.run_args import RunArguments

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--run_args", help="Path to RunArguments", type=str, required=True
    )

    args = parser.parse_args()

    with open(args.run_args, "r") as f:
        raw = f.read()
        run_args = RunArguments.parse_raw(raw)

    itcib = OntologyPopulationRankedRetrievalDataset(
        in_file=run_args.input,
        system_message=run_args.system_message,
        user_prompt_template=run_args.user_prompt_template,
        regex=run_args.regex,
        examples_file=run_args.examples_file,
        llm_name=run_args.llm_name,
        **run_args.kwargs,
    )
    num_samples = len(itcib)
    itcib = iter(itcib)
    for i in range(num_samples):
        # print(i)
        inst, template, label = next(itcib)
        print(inst)
        for r in template:
            print("-" * 40)
            # print(r["content"])
