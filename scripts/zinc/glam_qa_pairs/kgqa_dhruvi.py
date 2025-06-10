import pandas as pd
import google.generativeai as genai
import igraph as ig
import google.api_core.exceptions

# Ask for API key from user
api_key = input("Enter your Gemini API key: ")
genai.configure(api_key=api_key)

# Choose an available model
model = genai.GenerativeModel("gemini-1.5-flash-latest")  # Faster & optimized for token limits

import time

def is_mental_health_related(llm, batch):
    """ Ask Gemini Pro if a batch of entities is related to mental health. Retries on failure. """
    max_retries = 5
    delay = 5  # Initial wait time in seconds

    for attempt in range(max_retries):
        try:
            prompts = [
                f"Entity: {name}, Type: {etype}. Is this related to mental health? Answer 'yes' or 'no'."
                for name, etype in batch
            ]

            responses = llm.generate_content(prompts)
            return [r.text.strip().lower() == "yes" for r in responses]

        except google.api_core.exceptions.ResourceExhausted:
            print(f"API quota exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff (double the wait time)

    print("API is still exhausted after retries. Skipping batch.")
    return [False] * len(batch)  # Mark all as not related



def extract_N_m(G, nodes_df, batch_size=10):
    """
    Select a subset N_m from KG where elements are related to mental health 
    and have entity type {disease, symptom}.
    i think it can be done separately either using LLM or some keyword matching method
    """

    """ Find mental health-related nodes using Gemini Pro in batches. """
    
    mental_health_nodes = []
    batch = []

    for _, row in nodes_df.iterrows():
        batch.append((row["name"], row["type"]))

        if len(batch) >= batch_size:
            results = is_mental_health_related(model, batch)
            mental_health_nodes.extend([name for (name, _), is_related in zip(batch, results) if is_related])
            batch = []  # Reset batch

    # Process remaining nodes if any
    if batch:
        results = is_mental_health_related(model, batch)
        mental_health_nodes.extend([name for (name, _), is_related in zip(batch, results) if is_related])

    return mental_health_nodes


def extract_k_hop_subgraph(KG, N, k, N_m):
    """
    Extract the k-hop subgraph around node N, ensuring we do not revisit nodes from N_m.

    Parameters:
    - KG: igraph.Graph, the knowledge graph.
    - N: str, the starting node.
    - k: int, number of hops.
    - N_m: set, mental health-related nodes to avoid.

    Returns:
    - subgraph: igraph.Graph, extracted k-hop subgraph.
    """
    if N not in KG.vs["name"]:
        raise ValueError(f"Node {N} not found in the graph.")

    visited = set()
    queue = [(N, 0)]
    subgraph_nodes = set()

    while queue:
        node, depth = queue.pop(0)
        if depth > k or node in visited or node in N_m:
            continue
        visited.add(node)
        subgraph_nodes.add(node)

        neighbors = KG.neighbors(KG.vs.find(name=node), mode="all")  # Get all neighbors
        for neighbor in neighbors:
            neighbor_name = KG.vs[neighbor]["name"]
            if neighbor_name not in visited and neighbor_name not in N_m:
                queue.append((neighbor_name, depth + 1))

    # Extract subgraph with edges
    subgraph = KG.subgraph([KG.vs.find(name=n).index for n in subgraph_nodes])
    return subgraph


def encode_subgraph_to_text(subgraph):
    """
    Encode the k-hop subgraph into text using (node1, relation, node2) format.

    Args:
        subgraph (igraph.Graph): The extracted subgraph.

    Returns:
        list of str: Encoded text representation in "node1 relation node2" format.
    """
    text_representation = []
    
    for edge in subgraph.es:
        node1 = subgraph.vs[edge.source]["name"]
        node2 = subgraph.vs[edge.target]["name"]
        relation = edge["relation"] if "relation" in edge.attributes() else "is related to"

        # Format as text
        text_representation.append(f"{node1} {relation} {node2}")
    
    return text_representation


import re

def transform_to_human_readable(text_representation):
    """
    Transform text representations into human-readable form by replacing underscores, 
    fixing case, and improving formatting.

    Args:
        text_representation (list of str): List of "node1 relation node2" strings.

    Returns:
        list of str: Human-readable descriptions.
    """
    readable_texts = []
    
    for text in text_representation:
        # Replace underscores with spaces and fix case
        cleaned_text = re.sub(r"_", " ", text).strip().capitalize()
        readable_texts.append(cleaned_text)
    
    return readable_texts

import google.generativeai as genai

def summarize_with_LLM(text):
    """
    Use an LLM to summarize and refine text by removing redundancy.
    
    Args:
        text (str): The input text to be summarized.
    
    Returns:
        str: A refined and concise version of the text.
    """
    model = genai.GenerativeModel("gemini-1.5-pro")  # Adjust model as needed

    prompt = f"""
    Refine the following text by making it concise, removing redundancy, 
    and improving readability while keeping all important details:
    
    "{text}"
    
    Provide only the refined text as the response.
    """

    response = model.generate_content(prompt)
    return response.text.strip()


def generate_questions(text):
    """
    Generate structured questions from text using prompting techniques.

    Args:
        text (str): The input text from which to generate questions.

    Returns:
        list: A list of generated questions.
    """
    model = genai.GenerativeModel("gemini-1.5-pro")  # Adjust model as needed

    prompt = f"""
    Given the following text, generate a set of structured questions that assess 
    comprehension and understanding. Focus on key details, relationships, and 
    important facts:

    "{text}"

    Provide the questions as a numbered list.
    """

    response = model.generate_content(prompt)
    
    # Split response into a list of questions
    questions = response.text.strip().split("\n") if response.text else []
    
    return [q.strip() for q in questions if q.strip()]



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

# sample answer nodes
# filter subgraphs