import sys
import time
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

csv_df = Path("./data/mhqa-rem-1714.csv")
if not csv_df.exists():
    raise FileNotFoundError(f"File {csv_df} not found. Please check the path.")
df = pd.read_csv(csv_df)
outputs_folder = Path("./outputs")
outputs_folder.mkdir(parents=True, exist_ok=True)
csv_writer = csv.writer(open(outputs_folder / f"{csv_df.stem}-outputs-{timestring()}.csv", "w"))
csv_writer.writerow(['index', 'question', 'explanation', 'reasoning_paths'])

ner = OpenAINER()
retriever = MainSubgraphRetrieval()
pruner = PageRankPruner()
reasoning_path_generator = ShortestPathReasoningPathGenerator()
explainer = MainExplainer()

def pipeline(question, answer, options):
    logger.info(f"Running pipeline For:")
    logger.info(f"QUESTION: {question}")
    logger.info(f"ANSWER: {answer}")
    logger.info(f"OPTIONS: {options}")

    question_entities, context_entities, answer_entities = ner.run(question, " ".join(options), answer)

    question_graph, qNodes = retriever.run(question_entities)
    context_graph, cNodes= retriever.run(context_entities)
    answer_graph, aNodes = retriever.run(question_entities + answer_entities)

    pruned_question_graph = pruner.run(question_graph, qNodes, 20)
    pruned_context_graph = pruner.run(context_graph, cNodes, 20)
    pruned_answer_graph = pruner.run(answer_graph, qNodes + aNodes, 20)

    question_path = reasoning_path_generator.run(pruned_question_graph, qNodes)
    context_path = reasoning_path_generator.run(pruned_context_graph, cNodes)
    answer_path = reasoning_path_generator.run(pruned_answer_graph, aNodes)

    reasoning_paths = question_path + context_path + answer_path
    if not reasoning_paths:
        logger.warning("No reasoning paths found")
        graph_context_prefix = ""
        graph_context = ""
    else:
        graph_context_prefix = "The following are the reasoning paths found in the graph: "
        logger.info(f"{len(reasoning_paths)} reasoning paths found")
        graph_context = graph_reasoning_paths_to_text(reasoning_paths)
        logger.info(f"Reasoning Paths: \n{graph_context}")

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

startTime = time.time()
with ThreadPoolExecutor(max_workers=32) as pool:
    futures = []
    for i in range(0, len(df), batch_size):
        batch = df[i:i+batch_size]
        futures.append(pool.submit(batch_pipeline, batch))
    for fut in tqdm(futures):
        fut.result()

endTime = time.time()
logger.info(f"Completed {len(futures) * batch_size} items in {endTime - startTime} seconds")