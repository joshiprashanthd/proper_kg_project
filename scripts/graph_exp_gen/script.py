import sys
sys.path.append("/home/sracha/proper_kg_project")

from concurrent.futures import ThreadPoolExecutor
from src.components.ner import OpenAINER
from src.components.retrieval.subgraph import MainSubgraphRetrieval
from src.components.pruning import PageRankPruner
from src.components.reasoning_path_generation import ShortestPathReasoningPathGenerator
from src.components.explanation import MainExplainer
from src.utils import logger, graph_reasoning_paths_to_text
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import csv

def timestring():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

df = pd.read_csv("/home/sracha/proper_kg_project/data/mhqa/mhqa_gold_final.csv")
log_folder = Path("./logs")
log_folder.mkdir(parents=True, exist_ok=True)
csv_writer = csv.writer(open(log_folder / f"outputs-{timestring()}.csv", "w"))
csv_writer.writerow(['index', 'question', 'explanation', 'reasoning_paths'])

ner = OpenAINER()
retriever = MainSubgraphRetrieval()
pruner = PageRankPruner()
reasoning_path_generator = ShortestPathReasoningPathGenerator()
explainer = MainExplainer()

def pipeline(question, answer, options):
    logger.info(f"Running pipeline for Question: {question}\nAnswer: {answer}\nOptions: {options}")
    question_entities = ner.run(question)
    answer_entities = ner.run(answer)
    context_entities = ner.run(options)

    question_graph, qNodes = retriever.run(question_entities)
    context_graph, cNodes= retriever.run(context_entities)
    answer_graph, aNodes = retriever.run(question_entities + answer_entities)

    pruned_question_graph = pruner.run(question_graph, qNodes, 20)
    pruned_context_graph = pruner.run(context_graph, cNodes, 20)
    pruned_answer_graph = pruner.run(answer_graph, aNodes, 20)

    question_path = reasoning_path_generator.run(pruned_question_graph, qNodes)
    context_path = reasoning_path_generator.run(pruned_context_graph, cNodes)
    answer_path = reasoning_path_generator.run(pruned_answer_graph, aNodes)

    reasoning_paths = question_path + context_path + answer_path
    if not reasoning_paths:
        logger.info("No reasoning paths found")
        graph_context_prefix = ""
        graph_context = ""
    else:
        graph_context_prefix = "The following are the reasoning paths found in the graph: "
        graph_context = graph_reasoning_paths_to_text(reasoning_paths)

    qna_context_prefix = "Options:"
    qna_context = " ".join(options)

    explanation = explainer.run(
            question,
            answer, 
            qna_context_prefix, 
            qna_context, 
            graph_context_prefix, 
            graph_context
        )

    return explanation, graph_context

def batch_pipeline(batch):
    for i, row in batch.iterrows():
        question = row['question']
        answer = row['correct_option']
        options = [row['option1'], row['option2'], row['option3'], row['option4']]
        explanation, graph_context = pipeline(question, answer, options)
        csv_writer.writerow([i, question, explanation, graph_context])

batch_size = 10

with ThreadPoolExecutor(max_workers=16) as pool:
    futures = []
    for i in range(0, 20, batch_size):
        batch = df[i:i+batch_size]
        futures.append(pool.submit(batch_pipeline, batch))

    for fut in tqdm(futures):
        fut.result()