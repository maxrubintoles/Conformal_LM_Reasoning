import ast
import json
from collections import deque
import numpy as np

# makes dependency graph an array if not already
def to_valid_array(input_value):  
    # Check if the input is already a list  
    if isinstance(input_value, list):  
        return input_value  
      
    # If the input is a string, try to evaluate it to a Python object  
    if isinstance(input_value, str):  
        # Replace any newline characters to ensure it evaluates correctly  
        input_value = input_value.replace('\n', '').replace('\\n', '')  
        try:  
            # Use ast.literal_eval to safely evaluate the string  
            return ast.literal_eval(input_value)  
        except (ValueError, SyntaxError):  
            raise ValueError("Input string is not a valid Python literal.")  
      
    # If the input is neither a list nor a string, raise an error  
    raise TypeError("Input must be a string or a list.") 


'''
node scoring method
'''
def get_ancestors(G, node):
    n = len(G)
    children = []

    for j in range(n):
        if G[node][j] == 1:  # There is an edge from node j to node (indicating j is a child of node)
            children.append(j)
    return children

def risk_score(question, top_sort, beta=0):
    G = to_valid_array(question["dep_graph"])
    
    node_risks = {}
    
    # Compute the baseline risk scores for all nodes first
    baseline_risks = {
        i: 6 - 1 * float(question["claims"][i]["frequency-score"]) for i in top_sort
    }
    
    for i in top_sort:
        node_risks[i] = baseline_risks[i]
        
        # Find descendants of node i
        descendants = [j for j in range(len(G)) if G[j][i] == 1]
        
        if descendants:
            # Compute the baseline risks for all descendants
            descendant_risks = [baseline_risks[j] for j in descendants]
            
            # Calculate the median of the descendant risks
            median_descendant_risk = sorted(descendant_risks)[len(descendant_risks) // 2]
            
            # Update the node's risk score using the linear combination
            node_risks[i] = (1 - beta) * node_risks[i] + beta * median_descendant_risk
            
    return node_risks


'''
def risk_score(question, top_sort, beta = 0):
    G = to_valid_array(question["dep_graph"])
    node_risks = {}
    for i in top_sort:
        node_risks[i] = 6 - 1 * float(question["claims"][i]["frequency-score"])
        parent_adjusted_risks = [min(0, node_risks[j]) for j in range(len(G)) if G[i][j] == 1]
        if parent_adjusted_risks:
            node_risks[i] += beta * max(parent_adjusted_risks)
    return node_risks
'''

'''
topological sort
'''
def topological_sort(adj_list):
    
    in_degree = {i: 0 for i in range(len(adj_list))}  # Initialize all in-degrees to 0

    # Compute in-degrees of all nodes
    for i in range(len(adj_list)):
        for j in range(len(adj_list[i])):
            if adj_list[i][j] == 1:
                in_degree[i] += 1

    # Initialize a queue and enqueue all nodes with in-degree 0
    queue = deque([node for node in in_degree if in_degree[node] == 0])

    top_order = []

    while queue:
        node = queue.popleft()
        top_order.append(node)

        # Reduce in-degree of adjacent nodes by 1
        for i in range(len(adj_list[node])):
            if adj_list[i][node] == 1:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    
    return top_order


'''
i/p: annotated question w/ graph
o/p: list of (at most) n + 1 subgraphs, each w/ heuristic risk
'''

def simult_search(question, noise, ind = False, risk = 'freq', beta = None):
    # in case no beta is passed, set it to 0
    if not beta:
        beta = 0

    n = len(question['claims'])
    # make sure graph parseable as an array
    adj_list = to_valid_array(question["dep_graph"])
    
    if ind:
        adj_list = [[0 for _ in range(n)] for _ in range(n)]

    subgraphs = {}
    subgraphs[0] = [[], noise - 1000]

    top_order = topological_sort(adj_list)
    
    node_scores = risk_score(question, top_order, beta)
    
    # add noise
    for i in range(len(node_scores)):
        node_scores[i] += noise

    # compute scores
    threshes = []
    for i in range(len(question["claims"])):
        # convert from confidence to risk
            if node_scores[i] not in threshes:
                threshes.append(node_scores[i])
        
    threshes.sort()

    k = 1

    for thresh in threshes:
        
        candidates = [i for i in range(len(question["claims"])) if node_scores[i] <= thresh]
        
        # topological sort of included idxs
        order_index = {node: i for i, node in enumerate(top_order)}
        
        candidates.sort(key=order_index.get)

        # iteratively remove nodes that have ancestors not included

        included = []

        while candidates:
            node = candidates.pop(0)
            # if all ancestors included, add node to included
            ancestors = [i for i in range(len(adj_list)) if adj_list[node][i] == 1]
            should_include = True
            for a in ancestors:
                if a not in included:
                    should_include = False
                    break
            if should_include:
                included.append(node)
    
        subgraphs[k] = [included, thresh + noise]
        k += 1

    return subgraphs


if __name__ == "__main__":
    file_path = 'data/MATH_annotated.json'

    with open(file_path, 'r') as file:
        try:
            # Parse the JSON content
            data = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
    
    questions = data.get("data", [])

    question = questions[0]

    beta = 0.2

    top_sort = topological_sort(question["dep_graph"])

    print(risk_score(question, top_sort, beta = beta))


