import sys
import os
import copy
from openai import OpenAI
import numpy as np
import json
import matplotlib.pyplot as plt
import math
from src.helpers.non_conformity import r_score, annotate, highest_risk_graph
import random
import numpy as np
from src.helpers.dependent_cp.greedy import greedy_search
from src.helpers.dependent_cp.simult import simult_search, to_valid_array

"""
i/p: risk scores
o/p: risk quantile
"""


def compute_quantile(scores, alpha):
    n = len(scores)
    scores = [-num for num in scores]
    scores.sort()

    index = math.ceil((n + 1) * (1 - alpha)) - 1

    # in case sample size is too small, must simply omit every claim
    if index > (n - 1):
        return -1000

    return -scores[index]


"""
i/p: method, alphas, ip filepath
o/p: realized factuality for each alpha
"""


def calibration(method, alphas, in_path):

    with open(in_path, "r") as file:
        try:
        # Parse the JSON content
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")

        questions = data.get("data", [])

    # oracle method (n * alpha rounded down errors)
    if method[2] == "oracle":
        n = len(questions)
        error_rates = []
        for alpha in alphas:
            error_rates.append(float((math.floor(alpha * n))) / n)
        fact_rates = [1 - error for error in error_rates]
        return fact_rates, np.zeros(len(alphas))

    seed = 42
    N = 100
    random.seed(seed)
    noise = np.random.normal(0, 1, len(questions))

    error_rates = []
    std_errs = []

    for alpha in alphas:

        alpha_error = []

        for i in range(N):
            # reload file to account for new manual annotations
            if i >= 1:
                with open(in_path, "r") as file:
                    try:
                    # Parse the JSON content
                        data = json.load(file)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON: {e}")

                    questions = data.get("data", [])
                
            partition = len(questions) // 2

            sample_error = []

            questions_copy = copy.deepcopy(questions)
            random.shuffle(questions_copy)

            # randomly assort data
            calibration_set, validation_set = (
                questions_copy[:partition],
                questions_copy[partition:],
            )

            # compute quantile
            if len(method) == 6:
                mult = method[5]
            else:
                mult = 0

            calibration_scores = [
                r_score(q, n, method[2], method[3], beta=mult, in_path = in_path)
                for q, n in zip(calibration_set, noise[:partition])
            ]
            quantile = compute_quantile(calibration_scores, alpha)

            # validate error
            for q, n in zip(validation_set, noise[partition:]):
                U_filt = highest_risk_graph(quantile, n, q, method[2], beta=mult)
                if annotate(U_filt, q, method[4],filepath=in_path) == 0:
                    sample_error.append(1)
                else:
                    sample_error.append(0)

            alpha_error.append(sum(sample_error) / len(validation_set))

        error_rates.append(sum(alpha_error) / N)
        std_errs.append(get_std_error(alpha_error))

    fact_rates = [1 - error for error in error_rates]

    return fact_rates, std_errs


"""
i/p: method, alphas, ip filepath
o/p: claims retained, realized factuality for each alpha
"""


def validation(method, alphas, in_path):
    with open(in_path, "r") as file:
        try:
            # Parse the JSON content
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")

    questions = data.get("data", [])
    
    n = len(questions)
    if method[2] == "oracle":

        most_true = {}

        for i in range(len(questions)):
            max_claims = 0
            for graph in questions[i]["graph_annotations"]["y"]:
                claims = 0
                for item in graph:
                    claims += item
                if claims > max_claims:
                    max_claims = claims
            most_true[i] = max_claims / len(questions[i]["claims"])

        most_true_sorted = dict(sorted(most_true.items(), key=lambda item: item[1]))

        # vals to return
        claims_retained = []
        realized_fact = []

        for alpha in alphas:
            alpha_retained = []

            false_num = math.floor(alpha * n)  # number of items to select

            realized_fact.append((n - false_num) / n)

            idx = 0
            for key, value in most_true_sorted.items():
                idx += 1
                if idx <= false_num:
                    alpha_retained.append(1)
                else:
                    alpha_retained.append(value)

            claims_retained.append(sum(alpha_retained) / len(alpha_retained))

        return realized_fact, claims_retained, np.zeros(len(alphas))

    seed = 42
    np.random.seed(seed)
    noise = np.random.normal(0, 1, len(questions))
    noise = noise.tolist()

    realized_fact = []
    claims_retained = []

    for alpha in alphas:

        alpha_error = []
        alpha_claims = []
        std_errs = []

        for i in range(len(questions)):

            # 49-1 split
            calibration_set, validation_set = (
                questions[:i] + questions[i + 1 :],
                questions[i],
            )
            noise_calib = noise[:i] + noise[i + 1 :]

            # compute quantile
            if len(method) == 6:
                mult = method[5]
            else:
                mult = 0
            calibration_scores = [
                r_score(q, n, method[2], method[3], beta=mult, in_path=in_path)
                for q, n in zip(calibration_set, noise_calib)
            ]
            quantile = compute_quantile(calibration_scores, alpha)

            U_filt = highest_risk_graph(
                quantile, noise[i], validation_set, method[2], beta=mult
            )
            if annotate(U_filt, validation_set, method[4], in_path) == 0:
                alpha_error.append(1)
            else:
                alpha_error.append(0)

            alpha_claims.append(len(U_filt) / len(validation_set["claims"]))

        std_errs.append(get_std_error(alpha_claims))

        realized_fact.append(1 - sum(alpha_error) / len(questions))
        claims_retained.append(sum(alpha_claims) / len(alpha_claims))

    return realized_fact, claims_retained, std_errs


def get_plots(methods, alphas_calib, alphas_valid, in_path, out_path):

    with open(in_path, "r") as file:
        try:
            # Parse the JSON content
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")

    questions = data.get("data", [])
    
    n = len(questions)

    # store calibration results for each method
    calib = {}

    # store validation results for each method
    valid = {}

    # get data for each method
    for method in methods:
        calib[method[0]] = calibration(method, alphas_calib, in_path)
        valid[method[0]] = validation(method, alphas_valid, in_path)

    """
    Calibration Plot
    """

    plt.figure()

    # Convert error to factuality
    target_fact_calib = [1 - alpha for alpha in alphas_calib]

    # Initialize lists to capture bounds
    all_x_values = []
    all_y_values = []

    # Collect data points for axis scaling
    for method in methods:
        # Extract color and name
        line_color = method[1]
        name = method[0]

        # Add the plotted data
        plt.plot(
            target_fact_calib, calib[name][0], label=name, linewidth=2, color=line_color
        )

        # Accumulate data points for axis scaling
        all_x_values.extend(target_fact_calib)
        all_y_values.extend(calib[name][0])

    # Compute dynamic bounds based on the data range
    x_min, x_max = min(all_x_values), max(all_x_values)
    x = np.linspace(x_min, x_max, 100)

    conformal_bound1 = x  # Identity line
    conformal_bound2 = x + 1 / ((n // 2) + 1)  # Shifted bound

    # Include conformal bounds in scaling
    all_x_values.extend(x)
    all_y_values.extend(conformal_bound1)
    all_y_values.extend(conformal_bound2)

    # Plot conformal bounds
    plt.plot(x, conformal_bound1, "k--", label="Conformal Bounds")
    plt.plot(x, conformal_bound2, "k--")

    # Add labels and title
    plt.xlabel("Target Factuality (1 - alpha)")
    plt.ylabel("Realized Coherent Factuality")
    plt.title("Calibration Plot")

    # Set the axis limits based on the data range with a small margin
    x_min, x_max = min(all_x_values), max(all_x_values)
    y_min, y_max = min(all_y_values), max(all_y_values)

    margin_x = 0.05 * (x_max - x_min)  # Add 5% margin to x-axis
    margin_y = 0.05 * (y_max - y_min)  # Add 5% margin to y-axis

    plt.xlim(x_min - margin_x, x_max + margin_x)
    plt.ylim(y_min - margin_y, y_max + margin_y)

    # Add the legend and position it below the plot
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    # Adjust the layout to ensure the legend doesn't overlap with the plot
    plt.tight_layout()

    out_file = f"{out_path}/calib_plot.png"
    plt.savefig(out_file, bbox_inches="tight")

    """
    Validation Plot
    """

    plt.figure()

    target_fact_valid = [1 - alpha for alpha in alphas_valid]

    for method in methods:

        # color given in m
        line_color = method[1]

        name = method[0]

        realized_fact = valid[name][0]
        percent_claims = valid[name][1]

        err = valid[name][2]

        # plt.plot(target_fact_valid, percent_claims, label=name, linewidth = 1, color = line_color)
        # plt.errorbar(target_fact_valid, percent_claims, yerr=err, linewidth = 2, color = line_color)

        plt.plot(
            realized_fact, percent_claims, label=name, linewidth=1, color=line_color
        )
        plt.errorbar(
            realized_fact, percent_claims, yerr=err, linewidth=2, color=line_color
        )

    # Add labels and title
    # plt.xlabel('1 - Alpha')
    plt.xlabel("Coherent Factuality")
    plt.ylabel("Percent of Claims Retained")

    plt.title("Claims Retained vs. Coherent Factuality")
    # plt.title('Claims Retained vs. Target Factuality')

    # Add a legend
    plt.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=1
    )  # Adjust ncol for the number of columns

    # Show the plot
    plt.autoscale()
    out_file = f"{out_path}/valid_plot.png"
    plt.savefig(out_file, bbox_inches="tight")


def get_std_error(vals):
    return np.std(vals) * 1.96 / np.sqrt(len(vals))


if __name__ == "__main__":

    # directory of current script
    base_dir = os.getcwd()


    # construct relative path to 'data' directory
    data_dir = os.path.join(base_dir, "data")

    # prompt user for full path
    filename = input("Enter the filename (e.g., MATH_open_annotations.json): ").strip()
    
    # construct ip/op path
    in_path = os.path.join(data_dir, filename)
    out_path = os.path.join(base_dir, "out")

    
    # check if file exists
    if not os.path.exists(in_path):
        print(f"Error: The file '{filename}' does not exist in the 'data' directory.")
        exit(1)

    """
    
    methods format: [ [label], [color], [filtering method], [calib. annos], [valid. annos], [beta (default = 0)] ]
    
    """

    methods = []

    methods.append([f"Subgraph Filtering", "blue", "simult", "graph", "manual", 0])
    methods.append(
        [f"Descendants Weight Boosting", "red", "simult", "graph", "manual", 0.5]
    )
    methods.append([f"Baseline", "orange", "ind", "ind", "manual", 0])

    calib_alphas = array = np.arange(0.25, 0, -0.025)

    valid_alphas = np.linspace(0, 0.25, 13)[1:]

    get_plots(
        methods,
        calib_alphas,
        valid_alphas,
        in_path,
        out_path,
    )
