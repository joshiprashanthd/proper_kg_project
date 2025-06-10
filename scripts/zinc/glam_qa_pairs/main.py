# for all nodes in the KG, select a subset N_m where elements in N_m are related to mental health and the entity type of each element is from {disease, symptom}
# given a node, N from the set N_m and a factor k, we firstly subset the k-hop subgraph around N.
# For simplicity (and if needed), if we start with a particular node, and the 1-hop subgraph around it already encodes any other nodes belonging to N_m, then we do not consider those nodes again for generating a k-hop subgraph
# Encoding subgraph it into text: (1) Extract all nodes and their relations within the k-hop subgraph. (2) use relational groups to encode the graph structure into text (3) Transform node and edge labels into human-readable descriptions
# Summarize using LLMs to remove redudant information and refine text
# Generate questions from the text using structured prompting techniques

def extract_N_m(KG):
    """
    Select a subset N_m from KG where elements are related to mental health 
    and have entity type {disease, symptom}.
    i think it can be done separately either using LLM or some keyword matching method
    """
    N_m = {node for node in KG.nodes if node.type in {"disease", "symptom"} and node.is_mental_health_related()}
    return N_m


def extract_k_hop_subgraph(KG, N, k, N_m):
    """
    Extract the k-hop subgraph around node N, ensuring we do not revisit nodes from N_m.
    """
    visited = set()
    subgraph = Graph()
    queue = [(N, 0)]  # (node, depth)
    
    while queue:
        node, depth = queue.pop(0)
        if depth > k or node in visited:
            continue
        visited.add(node)
        
        for neighbor, relation in KG.get_neighbors(node):
            subgraph.add_edge(node, neighbor, relation)
            if neighbor not in visited and neighbor not in N_m:
                queue.append((neighbor, depth + 1))
    
    return subgraph


def encode_subgraph_to_text(subgraph):
    """
    Encode the k-hop subgraph into text using relational groups.
    """
    relational_groups = {}
    for node in subgraph.nodes:
        for neighbor, relation in subgraph.get_neighbors(node):
            if relation not in relational_groups:
                relational_groups[relation] = []
            relational_groups[relation].append((node, neighbor))
    
    text_representation = ""
    for relation, pairs in relational_groups.items():
        text_representation += f"The following entities are connected by {relation}: "
        text_representation += ", ".join([f"{n1} and {n2}" for n1, n2 in pairs]) + ".\n"
    
    return text_representation


def transform_to_human_readable(text_representation):
    """
    Transform node and edge labels into human-readable descriptions.
    """
    # Implement mapping from technical labels to descriptive labels
    transformed_text = text_representation.replace("may_cause", "may lead to")
    transformed_text = transformed_text.replace("related_to", "is associated with")
    # Add other transformations as needed
    return transformed_text


def summarize_with_LLM(text):
    """
    Use an LLM to summarize and refine text by removing redundancy.
    """
    refined_text = LLM.summarize(text)
    return refined_text


def generate_questions(text):
    """
    Generate structured questions from text using prompting techniques.
    """
    questions = LLM.generate_questions(text)
    return questions


def process_knowledge_graph(KG, k):
    """
    Full pipeline: Extract N_m, generate subgraphs, encode, summarize, and generate questions.
    """
    N_m = extract_N_m(KG)
    all_results = {}
    
    for N in N_m:
        subgraph = extract_k_hop_subgraph(KG, N, k, N_m)
        text = encode_subgraph_to_text(subgraph)
        human_readable_text = transform_to_human_readable(text)
        summary = summarize_with_LLM(human_readable_text)
        questions = generate_questions(summary)
        
        all_results[N] = {"summary": summary, "questions": questions}
    
    return all_results
