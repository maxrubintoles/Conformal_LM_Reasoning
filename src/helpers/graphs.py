import json
import os
import copy
from tqdm import tqdm
import re

# GPT Graph Generation Prompt

'''

graph_few_shot = """I'm going to give you a question and a series of subclaims in response to the question.
I want you to create a dependency graph to represent the relationships between subclaims.
The set of vertices should be the set of subclaims. Then, if a subclaim "a" relies on another subclaim "b" to be considered true, include edge (b, a) in the graph (so a node's ancestors should contain all of its necessary assumptions).
Vertices that are "a priori" (e.g., assumptions given in the question, definitions, etc.), should not have ancestors.
Your final output will be an adjacency list.

Next, I'll give you some examples to make this clear.

Question: How many vertical asymptotes does the graph of $y=\\frac{x}{x^2+1}$ have?

Subclaim 1: A function has vertical asymptotes exactly where its denominator equals zero.
Subclaim 2: To solve for the vertical asymptotes of the function $y=\\frac{x}{x^2+1}$, we therefore must solve $x^2+1=0$.
Subclaim 3: For all real values of $x$, $x^2+1 > 0$
Subclaim 4: Thus, we conclude that the function $y=\\frac{x}{x^2+1}$ has no vertical asymptotes.

Desired Output:
[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,1,0]]

Commentary:

You should output an object like the one above without any other reasoning or formatting. In particular, you should output an array of n arrays, each of length n, where n is the number of subclaims. If subclaim j relies on the information from subclaim i, the jth array should have the ith entry = 1; otherwise this entry should be zero.
In this case, note that subclaim 1 does not have ancestors, because it does not require other steps to be justified (we assume common mathematical theorems, like the presence of vertical asymptotes when the denominator is zero, to be a priori). However, subclaim 2 relies on the conclusion of subclaim 1 since it sets the denominator equal to zero. Subclaim 3 implicitly relies on subclaim 2, since we derive this check from subclaim 2. Also, the final answer, subclaim 4, relies on combining information from both subclaims 2 and 3 (which describe the significance of the equation $x^2+1=0$ and its answer, respectively).
Also note that in generating this graph, we represent implicit relationships between claims: subclaim 4, for instance, doesn't cite subclaims 2 and 3 explicitly, but it certainly relies on their contents. For this reason, we put those edges in its adjacency list.
It is very important to represent all relationships in this way. In general, it is unlikely that a claim should be completely "floating" (not relied upon by or reliant upon another claim); in this case, it would not be contributing to the complete output.

By convention, we never include a claim in its own adjacency list (we don't consider a claim to rely on itself).

Here, we're interested in the dependency between claims, not just the correctness. For this reason, it's also important to represent these dependencies even in the case that an answer is wrong.

I'll give you another example below.

Question: Consider the function $y = x^2+2x+15$. What is the sum of the zeroes of this function?

Subclaim 1: The zeroes of a function are the x-values of its x-intercepts.
Subclaim 2: To find the zeroes of $y=x^2+2x+15$, we set the right hand side equal to $0$, writing $0=x^2+2x+15$.
Subclaim 3: To solve $0=x^2+2x+15$, we factor it as $0=(x+3)(x-5)$.
Subclaim 4: This means that the zeroes of $y=x^2+2x+15$ are $x=-3,5$.
Subclaim 5: We conclude that the sum of the zeroes of this function is $-3+5=2$.

Desired Output:
[[0,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]

Commentary:

Here, each claim simply relies on the previous claim. Importantly, subclaim 3 makes an algebraic error, incorrectly factoring as $0=(x+3)(x-5)$ instead of $0=(x-3)(x+5)$, which means the final answer is wrong. Even so, this subclaim relies on information from subclaim 2, and subclaim 4 relies on the conclusion from subclaim 3, so we represent these edges in our final output.
We are agnostic to correctness, and focus solely on the reliance between subclaims. If subclaim i makes use of subclaim j, even incorrectly, subclaim j should be an ancestor of subclaim i in our adjacency list.

Now, I'm going to give you another question and list of subclaims, as before. With all of this explanation in mind, I want you to output an adjacency list with no other reasoning.
"""

'''

# Open-source generation (Llama has more trouble generating dependency graphs, especially since it produces outputs with more subclaims)
graph_few_shot = prompt = """
You are a system designed to create dependency graphs for subclaims in response to a given question. Your output must strictly adhere to the following instructions:

1. Graph Description:
   - Represent the dependency relationships between subclaims as a directed graph.
   - Each subclaim is a vertex in the graph.
   - An edge (b → a) exists if subclaim "a" depends on subclaim "b."
   - Subclaims that are "a priori" (e.g., assumptions or definitions) should not have any ancestors.

2. Output Format:
   - Provide your graph as an adjacency list of size NUM × NUM, where NUM is the number of subclaims (this will be given at the beginning of the prompt).
   -A template adjacency list with all entries zero will be included at the end of the prompt as reference. Your output should match this structure, but you should replace 0s with 1s where relevant, as described below.
   - Each entry in the adjacency list is a list of n integers:
     - A value of 1 at position i in row j indicates that subclaim j depends on subclaim i.
     - A value of 0 indicates no dependency.
     - Ensure no claim depends on itself (diagonal entries must be 0).

3. Rules:
   - The adjacency list must be square, with n rows and n columns, where n is the exact number of subclaims provided.
   - Each row and column must be exactly n integers. Do not include extra rows, columns, or misaligned entries.
   - The output must consist solely of the adjacency list (e.g., [[0,1,0],[0,0,1],[0,0,0]]); do not include explanations, commentary, or any other formatting.

4. Dependencies:
   - Consider explicit and implicit dependencies between subclaims. For example, if subclaim j implicitly relies on subclaim i (even if not stated directly), include the edge (i → j) in the graph.
   - Always represent dependencies, even if the subclaims are incorrect or contain logical errors.

Examples:

- Input:
  Question: How many vertical asymptotes does the graph of y = x / (x^2 + 1) have?

  NUM = 4
  Subclaims:
  1. A function has vertical asymptotes exactly where its denominator equals zero.
  2. To solve for the vertical asymptotes of the function y = x / (x^2 + 1), we therefore must solve x^2 + 1 = 0.
  3. For all real values of x, x^2 + 1 > 0.
  4. Thus, we conclude that the function y = x / (x^2 + 1) has no vertical asymptotes.

  Template:
  [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

  Desired Output:
  [[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,1,0]]

- Input:
  Question: Consider the function y = x^2 + 2x + 15. What is the sum of the zeroes of this function?

  NUM = 5  
  Subclaims:
  1. The zeroes of a function are the x-values of its x-intercepts.
  2. To find the zeroes of y = x^2 + 2x + 15, we set the right-hand side equal to 0, writing 0 = x^2 + 2x + 15.
  3. To solve 0 = x^2 + 2x + 15, we factor it as 0 = (x+3)(x-5).
  4. This means that the zeroes of y = x^2 + 2x + 15 are x = -3, 5.
  5. We conclude that the sum of the zeroes of this function is -3 + 5 = 2.

  Template:
  [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

  Desired Output:
  [[0,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]

Now provide your adjacency list for the following question and subclaims:
"""

def parse_graph_output(output):
    """
    Parses the output from the model to extract the adjacency list.
    Ensures the adjacency list is a list of lists containing only 0 or 1.
    Replaces non-zero entries with 1 if invalid values are found.
    """
    # Remove any explanation or commentary before the list of lists
    output = re.sub(r'^[^\[]*\[', '[', output, flags=re.DOTALL)
    
    # Remove any newlines or extra spaces within the list of lists
    output = re.sub(r'\s+', '', output)
    
    try:
        # Evaluate the cleaned output to get the adjacency list
        adjacency_list = eval(output)
        
        # Ensure the adjacency list is a list of lists
        if all(isinstance(row, list) for row in adjacency_list):
            # Replace all non-zero entries with 1
            adjacency_list = [
                [0 if i == 0 else 1 for i in row] for row in adjacency_list
            ]
            return adjacency_list
    except (SyntaxError, ValueError):
        pass

    # Return None if validation fails
    return None


def query_model_system(client, system_prompt, user_prompt, model, max_tokens=2000, temperature=0, n_samples=1):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n_samples,
    )
    return completion.choices[0].message.content

def generate_template(num):
    """
    Generates a template adjacency list of size num × num with all zeros.
    """
    return [[0] * num for _ in range(num)]

def sequential_adjacency_list(num_nodes):
    """
    Creates a sequential adjacency list with directed edges.
    Example for num_nodes = 3:
    [[0, 1, 0],
     [0, 0, 1],
     [0, 0, 0]]
    """
    adj_list = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes - 1):
        adj_list[i][i + 1] = 1
    return adj_list

def is_cyclic(adj_list):
    """
    Detects if a graph is cyclic using Kahn's algorithm.
    Returns True if the graph is cyclic, False otherwise.
    """
    in_degree = {i: 0 for i in range(len(adj_list))}

    # Compute in-degrees of all nodes
    for i in range(len(adj_list)):
        for j in range(len(adj_list[i])):
            if adj_list[i][j] == 1:
                in_degree[j] += 1

    # Initialize a queue and enqueue all nodes with in-degree 0
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    visited_count = 0

    while queue:
        node = queue.popleft()
        visited_count += 1

        # Reduce in-degree of adjacent nodes by 1
        for i in range(len(adj_list[node])):
            if adj_list[node][i] == 1:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    # If all nodes are not visited, there's a cycle
    return visited_count != len(adj_list)

def add_graphs(questions, client, model):
    """
    Queries model to generate deducibility graph proxies for responses.
    If the generated graph is cyclic, replaces it with a sequential adjacency list.
    """
    updated_questions = copy.deepcopy(questions)
    
    for question in tqdm(updated_questions, desc="Processing Questions"):
        # Define the system prompt
        system_prompt = graph_few_shot
        
        # Define the user prompt
        num_subclaims = len(question["claims"])
        user_prompt = (
            "Question: " + question["prompt"] +
            f"\n\nNUM: {num_subclaims}\nSubclaims:"
        )
        
        # Add subclaims to the prompt
        for j, claim in enumerate(question["claims"], start=1):
            user_prompt += f"\n{j}. " + claim["subclaim"]
        
        # Generate the template adjacency list
        template = generate_template(num_subclaims)
        user_prompt += f"\n\nTemplate:\n{template}"
        
        max_attempts = 5
        attempts = 0
        valid_graph = False
        parsed_graph = None
        
        while attempts < max_attempts and not valid_graph:
            # Query the model
            dep_graph = query_model_system(
                client=client, 
                system_prompt=system_prompt, 
                user_prompt=user_prompt, 
                model=model, 
                max_tokens=1000, 
                temperature=0
            )
            
            # Parse and validate the graph
            parsed_graph = parse_graph_output(dep_graph)
            
            if parsed_graph and len(parsed_graph) == num_subclaims:
                valid_graph = True
            else:
                attempts += 1

        if valid_graph:
            if is_cyclic(parsed_graph):
                # Replace cyclic graph with sequential adjacency list
                question["dep_graph"] = sequential_adjacency_list(num_subclaims)
            else:
                # Store the valid acyclic graph
                question["dep_graph"] = parsed_graph
        else:
            print(f"Failed to generate valid graph after {max_attempts} attempts for question: {question['prompt']}")
            print(user_prompt)
            # Replace with sequential graph if unable to generate a valid graph
            question["dep_graph"] = sequential_adjacency_list(num_subclaims)
        
    return updated_questions

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def main(file_path):
    data = load_json(file_path)
    
    for question in data:
        print(f"Prompt: {question['prompt']}")
        print(f"Original Output: {question.get('original_output', 'N/A')}")
        print("Subclaims:")
        for i, claim in enumerate(question["claims"], start=1):
            print(f"{i}. {claim['subclaim']}")
        
        while True:
            user_input = input("Please enter the dependency graph as a list (or type 'skip' to skip): ")
            if user_input.lower() == 'skip':
                break
            
            try:
                gold_graph = json.loads(user_input)
                if isinstance(gold_graph, list):
                    question["gold_graph"] = gold_graph
                    save_json(data, file_path)
                    print("Graph saved successfully.")
                    break
                else:
                    print("Input is not a valid list. Please try again.")
            except json.JSONDecodeError:
                print("Invalid JSON format. Please try again.")

if __name__ == "__main__":
    file_path = "MATH_annotated.json"
    if os.path.exists(file_path):
        main(file_path)
    else:
        print(f"File {file_path} does not exist.")