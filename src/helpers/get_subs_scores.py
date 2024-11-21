import json
from math import ceil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.stats import rankdata
from helpers.sayless import (
    get_frequency_scores,
    get_subclaims,
    query_model,
)
from helpers.graphs import add_graphs
import random

CORRECT_ANNOTATIONS = ["Y", "S", "1"]


def get_legend_name(confidence_method):
    if (
        confidence_method == "frequency+gpt"
        or confidence_method == "frequency+gpt-ranking"
    ):
        return "Frequency"
    elif confidence_method == "baseline" or confidence_method == "baseline-ranking":
        return "Ordinal"
    elif confidence_method == "gpt" or confidence_method == "gpt-ranking":
        return "GPT-4 confidence"
    elif confidence_method == "random" or confidence_method == "random-ranking":
        return "Random"
    elif confidence_method == "optimal" or confidence_method == "optimal-ranking":
        return "Optimal"


def get_title_name(dataset_prefix):
    if dataset_prefix == "factscore":
        return "FActScore"
    elif dataset_prefix == "nq":
        return "NQ"
    elif dataset_prefix == "MATH":
        return "MATH"
    return dataset_prefix


def dump_claims(output_list, filename="claims.json"):
    """
    Dumps output_list into filename.
    """
    with open(filename, "w") as outfile:
        merged_json = {"data": output_list}
        json.dump(merged_json, outfile, indent=4)


def load_calibration(filename="claims.json"):
    """
    Reverse of dump_claims.
    """
    with open(filename, "r") as fopen:
        return json.load(fopen)["data"]


def get_ranking(entry, confidence_method, use_percent=True):
    """
    Returns the corresponding ranking scores from the raw scores of confidence_method.
    """
    score_list = [
        -(subclaim[confidence_method + "-score"] + subclaim["noise"])
        for subclaim in entry["claims"]
    ]
    rankings = len(entry["claims"]) + 1 - rankdata(score_list, method="ordinal")
    if use_percent:
        rankings = rankings / len(entry["claims"])
    return rankings


def get_confidence(entry, method, openai_client, model):
    """
    Takes in an entry from {}_annotations.json and returns a list of confidence scores from method.
    """
    if method == "random":
        return [np.random.normal(0, 1) for subclaim in entry["claims"]]
    elif method == "baseline":
        return [
            len(entry["claims"]) - x for x in list(range(1, len(entry["claims"]) + 1))
        ]
    elif method == "gpt":
        return [float(subclaim["gpt-score"]) for subclaim in entry["claims"]]
    elif method == "frequency":
        return get_frequency_scores(
            openai_client, entry["claims"], entry["prompt"], 5, model
        )
    # This assumes frequency was already added.
    elif method == "frequency+gpt":
        return [
            subclaim["gpt-score"] + subclaim["frequency-score"]
            for subclaim in entry["claims"]
        ]
    elif method == "optimal":
        return [
            int(subclaim["manual_annotation"] in CORRECT_ANNOTATIONS)
            for subclaim in entry["claims"]
        ]
    # This assumes the corresponding raw scores were already added.
    elif method in [
        "random-ranking",
        "baseline-ranking",
        "gpt-ranking",
        "frequency-ranking",
        "frequency+gpt-ranking",
        "optimal-ranking",
    ]:
        return get_ranking(
            entry, method[:-8]
        )  # -8 is to remove '-ranking' from method.
    else:
        print(f"{method} method is not implemented.")


def add_scores(calibration_data, filename, confidence_methods, openai_client, model):
    """
    Adds noise (to break ties later) and scores for each method in confidence_methods to filename.
    """
    # Add a random draw to the data if it does not already exist
    if "noise" not in calibration_data[0]["claims"][0]:
        for entry in tqdm(calibration_data):
            for i, output in enumerate(entry["claims"]):
                output["noise"] = np.random.normal(0, 0.001)

        # Write to file if any modification was made.
        dump_claims(calibration_data, filename)

    # If confidence_method is not already computed, compute and add it to calibration_data.
    for confidence_method in confidence_methods:
        if confidence_method + "-score" not in calibration_data[0]["claims"][0]:
            print(f"Computing {confidence_method} method")
            for entry in tqdm(calibration_data):
                score_list = get_confidence(
                    entry, confidence_method, openai_client, model
                )
                for i, output in enumerate(entry["claims"]):
                    output[confidence_method + "-score"] = score_list[i]

        # Write to file if any modification was made.
        dump_claims(calibration_data, filename)

    return calibration_data


def generate_data(
    dataset_prefix,
    input_dataset,
    client,
    model,
    breakdown_prompt,
    confidence_methods_raw,
    confidence_methods_ranking,
    open_source = False,
    create_graphs = True
):
    """
    Performs the desired analysis for a given dataset.
    """
    # Generate outputs and subclaims if they do not exist, in {dataset_prefix}_subclaims.json file that
    # can be (manually) copied over to /data/{dataset_prefix}_annotations.json and then annotated.
    if open_source:
        dataset_prefix = dataset_prefix + "_open"

    if not os.path.exists(f"data/{dataset_prefix}_annotations.json"):
        print(
            f"Creating dataset for annotation. When done, please copy data/{dataset_prefix}_subclaims.json to data/{dataset_prefix}_annotations.json and annotate."
        )
        data = []
        # Generate outputs for each prompt
        for prompt in tqdm(input_dataset):

            output = query_model(client, prompt, model)

            # Extract subclaims. "annotation" field is automtically set to 'N'.
            subclaims = get_subclaims(
                client, output, model
            )

            claim_list = [
                {
                    "subclaim": subclaim["subclaim"],
                    "gpt-score": subclaim["gpt-score"],
                    "manual_annotation": "0",
                }
                for subclaim in subclaims
            ]
            data.append(
                {"prompt": prompt, "original-output": output, "claims": claim_list}
            )
        if create_graphs:
            data = add_graphs(data, client, model)

        dump_claims(data, f"data/{dataset_prefix}_subclaims.json")

    else:
        # Otherwise, get the annotated subclaims add all scores if they are not already there.
        if not os.path.exists(f"data/{dataset_prefix}_subclaims_with_scores.json"):
            print(
                f"Computing scores for subclaims. These will appear in data/{dataset_prefix}_subclaims_with_scores.json"
            )
            calibration_data = load_calibration(
                f"data/{dataset_prefix}_annotations.json"
            )
            print(dataset_prefix)
            # print(calibration_data)
            add_scores(
                calibration_data,
                f"data/{dataset_prefix}_subclaims_with_scores.json",
                confidence_methods_raw + confidence_methods_ranking,
                client,
                model,
            )

        calibration_data = load_calibration(
            f"data/{dataset_prefix}_subclaims_with_scores.json"
        )