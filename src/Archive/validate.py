import sys
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
i/p: questions, method, alphas
o/p: realized factuality for each alpha
"""


def calibration(questions, method, alphas):
    if method[2] == "oracle":
        n = len(questions)
        error_rates = []
        for alpha in alphas:
            error_rates.append(float((math.floor(alpha * n))) / n)
        fact_rates = [1 - error for error in error_rates]
        return fact_rates, np.zeros(len(alphas))

    seed = 42
    N = 1
    random.seed(seed)
    noise = np.random.normal(0, 1, len(questions))

    error_rates = []
    std_errs = []

    for alpha in alphas:

        alpha_error = []

        for _ in range(N):

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
                r_score(q, n, method[2], method[3], beta=mult)
                for q, n in zip(calibration_set, noise[:partition])
            ]
            quantile = compute_quantile(calibration_scores, alpha)

            # validate error
            for q, n in zip(validation_set, noise[partition:]):
                U_filt = highest_risk_graph(quantile, n, q, method[2], beta=mult)
                if annotate(U_filt, q, method[4]) == 0:
                    sample_error.append(1)
                else:
                    sample_error.append(0)

            alpha_error.append(sum(sample_error) / len(validation_set))

        error_rates.append(sum(alpha_error) / N)
        std_errs.append(get_std_error(alpha_error))

    fact_rates = [1 - error for error in error_rates]

    return fact_rates, std_errs


"""
i/p: questions, method, alphas
o/p: claims retained, realized factuality for each alpha
"""


def validation(questions, method, alphas):
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
                r_score(q, n, method[2], method[3], beta=mult)
                for q, n in zip(calibration_set, noise_calib)
            ]
            quantile = compute_quantile(calibration_scores, alpha)

            U_filt = highest_risk_graph(
                quantile, noise[i], validation_set, method[2], beta=mult
            )
            if annotate(U_filt, validation_set, method[4]) == 0:
                alpha_error.append(1)
            else:
                alpha_error.append(0)

            alpha_claims.append(len(U_filt) / len(validation_set["claims"]))

            alpha_claims.append(len(U_filt) / len(validation_set["claims"]))

        std_errs.append(get_std_error(alpha_claims))

        realized_fact.append(1 - sum(alpha_error) / len(questions))
        claims_retained.append(sum(alpha_claims) / len(alpha_claims))

    return realized_fact, claims_retained, std_errs


def get_plots(data, methods, alphas_calib, alphas_valid, file_path):

    # store calibration results for each method
    calib = {}

    # store validation results for each method
    valid = {}

    # get data for each method
    for method in methods:
        calib[method[0]] = calibration(data, method, alphas_calib)
        valid[method[0]] = validation(data, method, alphas_valid)

    """
    Calibration Plot
    """

    plt.figure()

    # plot conformal bounds
    x = np.linspace(0, 1, 100)
    plt.plot(x, x, "k--", label="Conformal Bounds")
    plt.plot(x, x + 1 / ((len(data) // 2) + 1), "k--")

    # convert error to factuality

    target_fact_calib = [1 - alpha for alpha in alphas_calib]

    for method in methods:

        # color given in m
        line_color = method[1]

        name = method[0]

        plt.plot(
            target_fact_calib, calib[name][0], label=name, linewidth=2, color=line_color
        )

        # plt.errorbar(desired_fact, calib[name][0], yerr=calib[name][1], linewidth = 2, color = line_color)

    # Add labels and title
    plt.xlabel("Target Factuality (1 - alpha)")
    plt.ylabel("Realized Factuality")
    plt.title("Calibration Plot")

    # Show the plot
    plt.autoscale()
    out_file = f"{file_path}/calib_plot.png"
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
    out_file = f"{file_path}/valid_plot.png"
    plt.savefig(out_file, bbox_inches="tight")


def get_std_error(vals):
    return np.std(vals) * 1.96 / np.sqrt(len(vals))


if __name__ == "__main__":
    file_path = "data/MATH_open_subclaims_with_scores.json"

    with open(file_path, "r") as file:
        try:
            # Parse the JSON content
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")

    questions = data.get("data", [])

    methods = []

    # betas = np.arange(0, 1.1, 0.1)

    # colors = plt.cm.Blues(0.5 + 0.5 * (betas - np.min(betas)) / (np.max(betas) - np.min(betas)))

    methods.append([f"beta = 0", "blue", "simult", "graph", "graph", 0])
    methods.append([f"beta = 0.5", "green", "simult", "graph", "graph", 0.5])

    calib_alphas = array = np.arange(0.25, 0, -0.025)

    valid_alphas = np.linspace(0, 0.25, 13)[1:]

    get_plots(
        questions,
        methods,
        calib_alphas,
        valid_alphas,
        "/Users/maxonrubin-toles/Desktop/Conformal/Open_Source/out",
    )


"""


i/p: risk scores
o/p: risk quantile

def compute_quantile(scores, alpha):
    n = len(scores)
    scores = [-num for num in scores]
    scores.sort()

    index = ((math.ceil((n + 1) * (1 - alpha)) - 1))

    # in case sample size is too small, must simply omit every claim
    if index > (n-1):
        return -1000
    
    return -scores[index]


i/p: questions, method, alphas
o/p: realized factuality for each alpha

def calibration(questions, method, alphas):
    if method[2] == "oracle":
        n = len(questions)
        error_rates = []
        for alpha in alphas:
            error_rates.append(float((math.floor(alpha * n)))/n)
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
        
        for _ in range(N):
            
            partition = len(questions) // 2
            
            sample_error = []
            
            questions_copy = copy.deepcopy(questions)
            random.shuffle(questions_copy)

            # randomly assort data
            calibration_set, validation_set = questions_copy[:partition], questions_copy[partition:]

            # compute quantile
            if len(method) == 6:
                mult = method[5]
            else:
                mult = 0

            calibration_scores = [r_score(q, n, method[2], method[3], beta = mult) for q, n in zip(calibration_set, noise[:partition])]
            quantile = compute_quantile(calibration_scores, alpha)

            # validate error
            for q, n in zip(validation_set, noise[partition:]):
                U_filt = highest_risk_graph(quantile, n, q, method[2], beta = mult)
                if annotate(U_filt, q, method[4]) == 0:
                    sample_error.append(1)
                else:
                    sample_error.append(0)

            alpha_error.append(sum(sample_error) / len(validation_set))
            
        error_rates.append(sum(alpha_error)/N)
        std_errs.append(get_std_error(alpha_error))

    fact_rates = [1 - error for error in error_rates]
    
    return fact_rates, std_errs
    

i/p: questions, method, alphas
o/p: claims retained, realized factuality for each alpha

def validation(questions, method, alphas):
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
            most_true[i] = max_claims/len(questions[i]["claims"])

        most_true_sorted = dict(sorted(most_true.items(), key=lambda item: item[1]))
        
        # vals to return
        claims_retained = []
        realized_fact = []

        for alpha in alphas:
            alpha_retained = []

            false_num = math.floor(alpha * n)  # number of items to select

            realized_fact.append((n - false_num)/n)

            idx = 0
            for key, value in most_true_sorted.items():
                idx += 1
                if idx <= false_num:
                    alpha_retained.append(1)
                else:
                    alpha_retained.append(value)

            claims_retained.append(sum(alpha_retained)/len(alpha_retained))
            
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
            calibration_set, validation_set = questions[:i] + questions[i+1:], questions[i]
            noise_calib = noise[:i] + noise[i+1:]

            # compute quantile
            if len(method) == 6:
                mult = method[5]
            else: mult = 0
            calibration_scores = [r_score(q, n, method[2], method[3], beta = mult) for q, n in zip(calibration_set, noise_calib)]
            quantile = compute_quantile(calibration_scores, alpha)

            
            U_filt = highest_risk_graph(quantile, noise[i], validation_set, method[2], beta = mult)
            if annotate(U_filt, validation_set, method[4]) == 0:
                    alpha_error.append(1)
            else:
                alpha_error.append(0)
            
            alpha_claims.append(len(U_filt)/len(validation_set["claims"]))

            alpha_claims.append(len(U_filt)/len(validation_set["claims"]))

        std_errs.append(get_std_error(alpha_claims))

        realized_fact.append(1 - sum(alpha_error)/len(questions))
        claims_retained.append(sum(alpha_claims)/len(alpha_claims))
    
    return realized_fact, claims_retained, std_errs

def get_plots(data, methods, alphas_calib, alphas_valid, file_path):
    # store calibration results for each method
    calib = {}

    # store validation results for each method
    valid = {}

    # get data for each method
    for method in methods:
        calib[method[0]] = calibration(data, method, alphas_calib)
        valid[method[0]] = validation(data, method, alphas_valid)
    
    
    Calibration Plot
    
    
    plt.figure()

    # plot conformal bounds
    x = np.linspace(0, 1, 100)
    plt.plot(x, x, 'k--', label='Conformal Bounds')
    plt.plot(x, x + 1/((len(data) // 2)+1), 'k--')
    
    # convert error to factuality

    target_fact_calib = [1 - alpha for alpha in alphas_calib]

    for method in methods:
        
        # color given in m
        line_color = method[1]

        name = method[0]

        plt.plot(target_fact_calib, calib[name][0], label=name, linewidth = 2, color = line_color)
        print(calib[name][0])
    
        # plt.errorbar(desired_fact, calib[name][0], yerr=calib[name][1], linewidth = 2, color = line_color)

    # Add labels and title
    plt.xlabel('Target Factuality (1 - alpha)')
    plt.ylabel('Realized Factuality')
    plt.title('Calibration Plot')
    

    # Show the plot
    plt.autoscale()
    out_file = f"{file_path}/calib_plot.png"
    plt.savefig(out_file, bbox_inches='tight')
        
    
    Validation Plot
    

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
        
        plt.plot(realized_fact, percent_claims, label=name, linewidth = 1, color = line_color)
        plt.errorbar(realized_fact, percent_claims, yerr=err, linewidth = 2, color = line_color)

    # Add labels and title
    # plt.xlabel('1 - Alpha')
    plt.xlabel('Coherent Factuality')
    plt.ylabel('Percent of Claims Retained')
    

    plt.title('Claims Retained vs. Coherent Factuality')
    # plt.title('Claims Retained vs. Target Factuality')

    # Add a legend
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=1)  # Adjust ncol for the number of columns

    # Show the plot
    plt.autoscale()
    out_file = f"{file_path}/valid_plot.png"
    plt.savefig(out_file, bbox_inches='tight')
    
def get_std_error(vals):
    return np.std(vals) * 1.96 / np.sqrt(len(vals))


if __name__ == "__main__":
    file_path = 'data/MATH_open_subclaims_with_scores.json'

    
    method format:
    [ [label] , [color] , [filtering method] , [calib_anno] , [valid_anno] , [] ]

    -filtering method

    
    with open(file_path, 'r') as file:
        try:
            # Parse the JSON content
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    
    questions = data.get("data", [])
    
    methods = []

    methods.append([f"beta = 0", "blue", "simult", "graph", "graph", 0])
    methods.append([f"beta = 0.5", "green", "simult", "graph", "graph", 0.5])

    calib_alphas = array = np.arange(0.25, 0, -0.025)

    valid_alphas = np.linspace(0, 0.25, 13)[1:]

    # find_disparities(questions, valid_alphas)
    
    get_plots(questions, methods, calib_alphas, valid_alphas, "/Users/maxonrubin-toles/Desktop/Conformal/Open_Source")


"""
