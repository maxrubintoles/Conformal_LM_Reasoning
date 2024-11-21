import io
import json
import os

from openai import OpenAI
from together import Together
import numpy as np

from helpers.get_subs_scores import generate_data
from datasets import load_dataset

CONFIDENCE_METHODS_RAW = [
    "random",
    "baseline",
    "gpt",
    "frequency",
    "frequency+gpt",
    "optimal",
]
CONFIDENCE_METHODS_RANKING = [
    "random-ranking",
    "baseline-ranking",
    "gpt-ranking",
    #   "frequency-ranking",
    "frequency+gpt-ranking",
    "optimal-ranking",
]

MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
REQUIRED_FRAC_CORRECT = 1
BREAKDOWN_PROMPT = "Please breakdown the following input into a set of small, independent claims (make sure not to add any information), and return the output as a jsonl, where each line is {subclaim:[CLAIM], gpt-score:[CONF]}.\n The confidence score [CONF] should represent your confidence in the claim, where a 1 is obvious facts and results like 'The earth is round' and '1+1=2'. A 0 is for claims that are very obscure or difficult for anyone to know, like the birthdays of non-notable people. If the input is short, it is fine to only return 1 claim. The input is: "

DATASET_PREFIX = "MATH"
OPEN_SOURCE = "True"

if __name__ == "__main__":

    def math_merge_prompt(subclaims, prompt):
        claim_string = "\n".join(
            [subclaim["subclaim"] for i, subclaim in enumerate(subclaims)]
        )
        return f"You will get a math problem and a set of steps that are true. Construct an answer using ONLY the steps provided. Make sure to include all the steps in the answer, and do not add any additional steps or reasoning. These steps may not fully solve the problem, but merging them could assist someone in solving the problem. \n\nThe steps:\n{claim_string}\n\nThe math problem:\n{prompt}. Remember, do not do any additional reasoning, just combine the given steps."

    # Get key (check for open-source).
    if OPEN_SOURCE:
        API_KEY = os.environ.get("TOGETHER_API_KEY")
        CLIENT = Together(api_key=API_KEY)
    else:
        API_KEY = os.environ.get("OAI_KEY")
        CLIENT = OpenAI(api_key=API_KEY)
    
    if API_KEY is None:
        raise ValueError(
            "Key is not set - please set OAI_KEY to your OpenAI key if querying GPT; if querying an open-source model, set TOGETHER_API_KEY to your Together key (with command: export [PLATFORM]_KEY=[[PLATFORM]]_KEY])"
        )
    
    # Load questions from math
    dataset = load_dataset("competition_math")
    input_dataset = [question["problem"] for question in dataset["test"]][0:50]
    with io.open("out/MATH.json", "w") as fopen:
        json.dump(dataset["test"][0:50], fopen, indent=4)

    generate_data(
        DATASET_PREFIX,
        input_dataset,
        CLIENT,
        MODEL,
        BREAKDOWN_PROMPT,
        CONFIDENCE_METHODS_RAW,
        CONFIDENCE_METHODS_RANKING,
        open_source=OPEN_SOURCE,
        create_graphs=True
    )
