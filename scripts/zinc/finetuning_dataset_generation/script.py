import sys
sys.path.append("/home/sracha/proper_kg_project")

from datetime import datetime
import pandas as pd
import argparse
from pathlib import Path
from src.utils import logger
from concurrent.futures import ThreadPoolExecutor

from src.components.ner import OpenAINER
from src.components.retrieval.subgraph import MainSubgraphRetrieval
from src.components.pruning import PageRankPruner
from src.components.reasoning_path_generation import ShortestPathReasoningPathGenerator
from src.components.explanation import MainExplainer
from src.utils.graph_utils import graph_reasoning_paths_to_text
from tqdm import tqdm

def get_completed_batch_numbers(outputs_folder):
    completed_batches = set()
    for file in outputs_folder.glob("*.csv"):
        if file.name.startswith("batch_") and file.name.endswith(".csv"):
            batch_number = int(file.stem.split("_")[1])
            completed_batches.add(batch_number)
    return completed_batches

def pipeline(question, answer, options):
    question_entities, context_entities, answer_entities = ner.run(question, " ".join(options) if len(options) > 0 else "", answer)

    question_graph, qNodes = retriever.run(question_entities)
    context_graph, cNodes= retriever.run(context_entities)
    answer_graph, aNodes = retriever.run(question_entities + answer_entities)

    retrieved_question_paths = reasoning_path_generator.run(question_graph, qNodes)
    retrieved_context_paths = reasoning_path_generator.run(context_graph, cNodes)
    retrieved_answer_paths = reasoning_path_generator.run(answer_graph, qNodes + aNodes)

    retrieved_reasoning_paths = retrieved_question_paths + retrieved_context_paths + retrieved_answer_paths

    pruned_question_graph = pruner.run(question_graph, qNodes, 40)
    pruned_context_graph = pruner.run(context_graph, cNodes, 40)
    pruned_answer_graph = pruner.run(answer_graph, qNodes + aNodes, 40)

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
        graph_context = graph_reasoning_paths_to_text(reasoning_paths)

    qna_context_prefix = "Options:" if len(options) > 0 else ""
    qna_context = " ".join(options) if len(options) > 0 else ""

    explanation = explainer.run(
            question,
            answer,
            qna_context_prefix, 
            qna_context,
            graph_context_prefix, 
            graph_context
        )

    return explanation, graph_reasoning_paths_to_text(reasoning_paths), graph_reasoning_paths_to_text(retrieved_reasoning_paths)


def thread_func(batch: pd.DataFrame, batch_save_path):
    outputs = {
        "explanation": [],
        "reasoning_paths": [],
        "retrieved_reasoning_paths": []
    }
    for i, row in batch.iterrows():
        question = row['question']
        options = [row[f'option{i}'] for i in range(1, 5) if not pd.isna(row[f'option{i}'])]
        answer = row['answer']

        explanation, graph_context, retrieved_reasoning_paths = pipeline(question, answer, options)
        outputs["explanation"].append(explanation)
        outputs["reasoning_paths"].append(graph_context)
        outputs["retrieved_reasoning_paths"].append(retrieved_reasoning_paths)
    
    batch["explanation"] = outputs["explanation"]
    batch["reasoning_paths"] = outputs["reasoning_paths"]
    batch["retrieved_reasoning_paths"] = outputs["retrieved_reasoning_paths"]

    batch.to_csv(batch_save_path, index=False)

def run_threads(df, batches_folder, batch_size, workers):
    completed_batches = get_completed_batch_numbers(batches_folder)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_number = i // batch_size
            if batch_number in completed_batches:
                logger.info(f"Batch {batch_number} already completed. Skipping.")
                continue

            futures.append(executor.submit(thread_func, df[i:i + batch_size], batches_folder / f"batch_{batch_number}.csv"))

        for future in tqdm(futures):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset for fine-tuning.")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input CSV file.",
        required=True
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of workers for parallel processing."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for processing."
    )
   
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_folder_path = Path("./outputs")

    logger.info(f"Input file: {input_file}")
    logger.info(f"Output folder path: {output_folder_path}")

    df = pd.read_csv(args.input_file)

    timestring = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    batches_folder = Path(output_folder_path / (input_file.stem + f"_batches_{timestring}"))
    batches_folder.mkdir(parents=True, exist_ok=True)

    ner = OpenAINER()
    retriever = MainSubgraphRetrieval()
    pruner = PageRankPruner()
    reasoning_path_generator = ShortestPathReasoningPathGenerator()
    explainer = MainExplainer()

    run_threads(df, batches_folder, args.batch_size, args.workers)
