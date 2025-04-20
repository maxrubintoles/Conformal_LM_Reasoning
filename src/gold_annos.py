import os
import json
import itertools
from collections import deque
import sys


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def get_ancestors(graph, node):
    ancestors = set()

    def visit(i):
        row = graph[i]
        for j in range(len(row)):
            if row[j] == 1 and j not in ancestors:
                ancestors.add(j)
                visit(j)

    visit(node)
    return ancestors


def generate_legal_subgraphs(graph):
    nodes = range(len(graph))
    legal_subgraphs = []
    for size in range(1, len(nodes) + 1):
        for subset in itertools.combinations(nodes, size):
            if all(get_ancestors(graph, node).issubset(subset) for node in subset):
                legal_subgraphs.append([1 if i in subset else 0 for i in nodes])
    return legal_subgraphs


def topological_sort(adj_list, subgraph):
    in_degree = {i: 0 for i in range(len(adj_list))}

    # Calculate in-degrees based on Adj^T
    for i in range(len(adj_list)):
        if subgraph[i] == 1:
            for j in range(len(adj_list[i])):
                if adj_list[i][j] == 1 and subgraph[j] == 1:
                    in_degree[i] += 1

    queue = deque(
        [node for node in in_degree if in_degree[node] == 0 and subgraph[node] == 1]
    )
    top_order = []

    while queue:
        node = queue.popleft()
        top_order.append(node)

        for i in range(len(adj_list)):
            if adj_list[i][node] == 1 and subgraph[i] == 1:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    return top_order


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 -m src.gold_annos <first_question_index> <last_question_index>"
        )
        return

    first_question_index = int(sys.argv[1])
    last_question_index = int(sys.argv[2])

    file_name = input("Enter the file name (e.g., MATH_open_subclaims.json): ")
    file_path = f"data/{file_name}"
    key_file_path = "data/MATH_key.json"

    try:
        data = load_json(file_path)
        key_data = load_json(key_file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    questions = data["data"]

    for question_index in range(first_question_index, last_question_index + 1):
        question = questions[question_index]
        graph = question["dep_graph"]

        # Check if 'graph_annotations' key exists, if not, initialize it
        if "graph_annotations" not in question:
            question["graph_annotations"] = {"y": [[]], "n": []}
        else:
            if "y" not in question["graph_annotations"]:
                question["graph_annotations"]["y"] = [[]]
            if "n" not in question["graph_annotations"]:
                question["graph_annotations"]["n"] = []

        legal_subgraphs = generate_legal_subgraphs(graph)

        prompt = question["prompt"]
        solution = None

        i = 0
        for problem in key_data["problem"]:
            if problem == prompt:
                solution = key_data["solution"][i]
                break
            i += 1

        if solution is None:
            print(f"No solution found for prompt: {prompt}")
            continue

        for subgraph in legal_subgraphs:
            sorted_subgraph_indices = topological_sort(graph, subgraph)
            sorted_subclaims = [
                question["claims"][i]["subclaim"] for i in sorted_subgraph_indices
            ]
            print(f"Prompt: {prompt}")
            print()
            print(f"Solution: {solution}")
            print()
            print("Topologically sorted claims:")
            for subclaim in sorted_subclaims:
                print(f"> {subclaim}")
            print()

            while True:
                user_input = (
                    input("Is this a coherently factual output? (Y/N): ")
                    .strip()
                    .upper()
                )
                if user_input in ["Y", "N"]:
                    break
                else:
                    print("Invalid input. Please enter 'Y' or 'N'.")

            if user_input == "Y":
                question["graph_annotations"]["y"].append(subgraph)
            else:
                question["graph_annotations"]["n"].append(subgraph)

            print()

            # Save the annotation to the file each time one is completed
            save_json(data, file_path)

    # Rename the file to have suffix "annotations" instead of "subclaims"
    if file_path.endswith("subclaims.json"):
        new_file_path = file_path.replace("subclaims.json", "annotations.json")
        os.rename(file_path, new_file_path)
        print(f"File renamed to: {new_file_path}")


if __name__ == "__main__":
    main()
