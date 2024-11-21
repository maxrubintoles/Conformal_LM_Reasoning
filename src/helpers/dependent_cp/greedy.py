import ast
import copy

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
i/p: subgraph
o/p: subgraph with additional claim, cum. heuristic risk
'''
def greedy_step(question):
 
    # subclaims
    subclaims = question["claims"]
    
    # graph
    graph = question["dep_graph"]

    # make sure graph parseable as an array
    graph = to_valid_array(graph)
    
    # determine which nodes remain
    
    included = []
    not_included = []
    # index tracker
    i = 0
    for s in subclaims:
        if "manual_annotation" in s:
            if s["manual_annotation"] == "Added":
                included.append(i)
            else:
                not_included.append(i)
        else:
            not_included.append(i)
        i += 1
    
    # node options for greedy search
    legal_options = []
    
    for subc in not_included:
        subc_adjacency = graph[subc]
        subc_ancestors = []
        i = 0
        for item in subc_adjacency:
            if item == 1:
                subc_ancestors.append(i)
            i += 1
        
        legal = True

        for node in subc_ancestors:
            if node not in included:
                legal = False
        
        if legal:
            legal_options.append(subc)

    choice = None
    # freq. score always higher than this
    max_score = -1000

    for option in legal_options:
        if "frequency_score" in subclaims[option]:
            if subclaims[option]["frequency_score"] > max_score:
                max_score = subclaims[option]["frequency_score"]
                choice = option
                risk_modifier = (5 - max_score)

        else:
            # some annotations written with dash instead of underscore
            if subclaims[option]["frequency-score"] > max_score:
                max_score = subclaims[option]["frequency-score"]
                choice = option
                risk_modifier = (5 - max_score)
    
    question["claims"][choice]["manual_annotation"] = "Added"
    
    included.append(choice)  
            
    return (question, included, risk_modifier)

'''
i/p: annotated question w/ graph
o/p: list of n + 1 greedy subgraphs, each w/ heuristic risk
'''

def greedy_search(question, noise, cot = True):

    q = copy.deepcopy(question)
    n = len(q["claims"])

    greedy_graphs = {}
    # tie-breaking noise
    risk = noise
    greedy_graphs[0] = [[], risk]
    if cot:
        m = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(1, n):
            m[i][i-1] = 1
        
        q['dep_graph'] = m
    for i in range(n):
        old_q = q
        q, nodes, risk_mod = greedy_step(old_q)
        risk += risk_mod
        greedy_graphs[i + 1] = [nodes, risk]

    return greedy_graphs
