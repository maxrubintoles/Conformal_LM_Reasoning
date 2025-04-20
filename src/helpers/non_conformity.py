from src.helpers.dependent_cp.greedy import greedy_search
from src.helpers.dependent_cp.simult import (
    simult_search,
    topological_sort,
    to_valid_array,
)
import json


# makes dependency graph an array if not already
def to_valid_array(input_value):
    # Check if the input is already a list
    if isinstance(input_value, list):
        return input_value


def manual_anno(
    subgraph,
    question,
    in_path,
    calib_method="graph",
):

    # assume empty output to be correct
    if subgraph == []:
        return 1

    # make dependency graph a valid array
    dep_graph = to_valid_array(question["dep_graph"])

    # ensure nodes in numeric order to corr. with annotations
    subgraph.sort()

    # convert list of indices to form of manual annotations
    idxs = []
    for i in range(len(dep_graph)):
        if i in subgraph:
            idxs.append(1)
        else:
            idxs.append(0)

    # Ensure that "graph_annotations" exists in the question
    if "graph_annotations" not in question:
        question["graph_annotations"] = {"y": [], "n": []}

    # Check if the idxs already exist in the annotations
    if idxs in question["graph_annotations"]["y"]:
        return 1
    elif idxs in question["graph_annotations"]["n"]:
        return 0
    else:
        # The annotation doesn't exist, prompt the user

        # Topologically sort the entire dependency graph
        if calib_method == "graph":
            sorted_indices = topological_sort(dep_graph)
        else:
            sorted_indices = list(range(len(dep_graph)))

        # Output the prompt and corresponding sorted claims
        print()
        print(f"Prompt: {question['prompt']}")
        print()
        print("Subgraph (sorted topologically):")
        print()
        sorted_subgraph_claims = [
            question["claims"][i]["subclaim"] for i in sorted_indices if i in subgraph
        ]

        # Print the sorted claims corresponding to the subgraph
        for claim in sorted_subgraph_claims:
            print(f"> {claim}")

        print()
        # Ask user for input
        user_input = input(
            "Enter '1' if this subgraph is correct, '0' if it is incorrect: "
        )

        # Validate user input
        while user_input not in ["1", "0"]:
            print("Invalid input. Please enter '1' or '0'.")
            user_input = input(
                "Enter '1' if this subgraph is coherently factual, '0' if not: "
            )

        # Store the subgraph in the correct list based on user input
        if user_input == "1":
            question["graph_annotations"]["y"].append(idxs)
            print("Updated graph_annotations (y):", question["graph_annotations"])
            updated = True
        elif user_input == "0":
            question["graph_annotations"]["n"].append(idxs)
            print("Updated graph_annotations (n):", question["graph_annotations"])
            updated = True

        print(idxs)

        # If there was an update, write the changes back to the file
        if updated:
            try:
                with open(in_path, "r+") as file:
                    # Read the current data
                    questions = json.load(file)
                    data = questions["data"]
                    # Find the matching question based on the prompt
                    for q in data:
                        if q["prompt"] == question["prompt"]:
                            q["graph_annotations"] = question["graph_annotations"]
                            break

                    # Move the cursor to the beginning of the file and overwrite the content
                    file.seek(0)
                    json.dump(questions, file, indent=4)
                    file.truncate()  # Ensure the file is properly truncated if the new content is shorter

                print(f"Updated annotations saved to {in_path}.")
            except IOError as e:
                print(f"Error saving file: {e}")

        return 1 if user_input == "1" else 0


# check subgraph validity according to individual annotations (assuming validity of GPT graph)
def graph_anno(subgraph, question):
    # assume empty output to be correct
    if subgraph == []:
        return 1

    # make dependency graph a valid array
    dep_graph = to_valid_array(question["dep_graph"])

    # check whether each node has all its ancestors in the subgraph
    for node in subgraph:
        for ancestor in range(len(dep_graph)):
            if (dep_graph[node][ancestor] == 1) and (ancestor not in subgraph):
                return 0

    # Find all roots (nodes without ancestors)
    roots = [
        node
        for node in range(len(dep_graph))
        if all(dep_graph[node][i] == 0 for i in range(len(dep_graph)))
    ]

    # Check if each node has a path to a root
    visited = [False] * len(dep_graph)

    for root in roots:
        stack = [root]  # Start from the root

        while stack:
            node = stack.pop()
            visited[node] = True

            # Add unvisited children to the stack
            for i in range(len(dep_graph[node])):
                if dep_graph[i][node] == 1 and not visited[i]:
                    stack.append(i)

    # If any node in the subgraph is not visited, it doesn't have a path to a root
    if not all(visited[node] for node in subgraph):
        return 0

    # Check whether there's an individual "bad" annotation
    for node in subgraph:
        if question["claims"][node]["manual_annotation"] in {0, "0", "0.0", "N"}:
            return 0

    return 1


# consider subclaims to be unordered, check according to old annotations
def ind_anno(subclaims, question):
    # assume empty output to be correct
    if subclaims == []:
        return 1

    # check (old) annotations individually
    for i in subclaims:
        if question["claims"][i]["manual_annotation"] in {0, "0.0", "0", "N"}:
            return 0

    return 1


def annotate(subclaims, question, type, filepath=None):
    if type == "ind":
        return ind_anno(subclaims, question)
    elif type == "graph":
        return graph_anno(subclaims, question)
    elif type == "manual":
        return manual_anno(subclaims, question, filepath)
    """
    catch error so if there is no manual annotation, you're prompted for one, it's added to the file, and the function is called again
    """


def highest_risk_graph(tau, noise, question, method, beta=None):
    if method == "simult":
        U = simult_search(question, noise, beta=beta)
    elif method == "greedy":
        U = greedy_search(question, noise)
    elif method == "ind" or method == "ind_filtered":
        U = simult_search(question, noise, ind=True)

    select_graph = U[len(U) - 1][0]

    # iterate over each subgraph
    for i in range(len(U)):
        if U[i][1] > tau:
            idx = max(0, i - 1)
            select_graph = U[idx][0]
            break

    # post-hoc filtering
    if method == "ind_filtered":
        # get a topological sort
        adj_list = to_valid_array(question["dep_graph"])
        top_order = topological_sort(adj_list)

        # topological sort of included idxs
        order_index = {node: i for i, node in enumerate(top_order)}
        select_graph.sort(key=order_index.get)

        # iteratively remove nodes that have ancestors not included
        included = []
        while select_graph:
            node = select_graph.pop(0)
            # if all ancestors included, add node to included
            ancestors = [i for i in range(len(adj_list)) if adj_list[node][i] == 1]
            if all([i in included for i in ancestors]):
                included.append(node)
        select_graph = included

    # if risk never exceeded, return entire output
    return select_graph


"""
i/p: annotated question/answer pair
o/p: "permissible risk" threshold, 
"""


def r_score(
    question, noise, method="simult", anno_type="manual", beta=None, in_path=None
):
    if method == "ind_filtered":
        method = "ind"
    if method == "simult":
        U = simult_search(question, noise, beta=beta)
    elif method == "greedy":
        U = greedy_search(question, noise)
    elif method == "ind":
        U = simult_search(question, noise, ind=True)

    # Not sure what these methods do
    """
    elif method == "cot":
        U = greedy_search(question, noise, cot = True)
    elif method == 'adj_freq':
        U = simult_search(question, noise, risk = 'prop-freq')
    
    """

    if anno_type == "manual":

        for i in range(len(U)):
            subg = U[i]
            vertices = subg[0]
            risk = subg[1]

            if not manual_anno(vertices, question, in_path):
                return risk

        # more than the maximum realizable risk
        return 1000

    elif anno_type == "graph":
        for i in range(len(U)):
            subg = U[i]
            vertices = subg[0]
            risk = subg[1]
            if not graph_anno(vertices, question):
                return risk

        # more than the maximum realizable risk
        return 1000

    elif anno_type == "ind":
        for i in range(len(U)):
            subg = U[i]
            vertices = subg[0]
            # print(vertices)
            risk = subg[1]
            if not ind_anno(vertices, question):
                return risk

        # more than the maximum realizable risk
        return 1000
