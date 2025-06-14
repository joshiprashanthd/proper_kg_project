import sys
sys.path.append("/home/sracha/proper_kg_project")

import pandas as pd
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime

from src.components.explanation import TripletRAGExplainer, OneParagraphExplainer
from src.components.faiss_rag import FaissRag
from src.components.bm25_rag import BM25Rag
from src.components.faiss_triplets_rag import FaissTripletsRag
from src.components.bm25_triplets_rag import BM25TripletsRag
from src.components.conversations import MCQAConversation
from src.utils import merge_jsonl_files, get_completed_batch_numbers, keep_first_definition
from src.types import Conversation

CONTEXT_TEMPLATE = """Below is the additional context for query:
<rag_paragraphs>
{rag_results}
</rag_paragraphs>
"""

def build_context(results):
    return "\n\n".join([result.content for result in results])

def pipeline(question: str, options: list[str], answer: str):
    query = f"Question: {question}\nOptions: {options}\nAnswer: {answer}"
    # faiss_triplets_rag_docs = faiss_triplets_rag.run(query)
    # bm25_triplets_rag_docs = bm25_triplets_rag.run(query)
    faiss_triplets_rag_docs = []
    bm25_triplets_rag_docs = []
    
    # usefulness_triplets_docs = triplet_usefulness.run(query, keep_first_definition([t.content for t in faiss_triplets_rag_docs + bm25_triplets_rag_docs]))
    # usefulness_triplets_docs = [t for t in sorted(usefulness_triplets_docs, key=lambda t: t.score, reverse=True) if t.label == "Useful"]
    usefulness_triplets_docs = []
    
    # naive_rag_query = f"""{query}
    # {build_context(list(set(usefulness_triplets_docs)))}"""

    naive_rag_query = query
    
    faiss_rag_docs = faiss_rag.run(naive_rag_query) 
    bm25_rag_docs = bm25_rag.run(naive_rag_query)
    
    context = CONTEXT_TEMPLATE.format(
        rag_results=build_context(list(set(faiss_rag_docs + bm25_rag_docs))),
        triplets="\n\n".join(keep_first_definition([t.triplet for t in usefulness_triplets_docs]))
    )
    
    explanation = triplet_explainer.run(query, context)
    one_paragraph_explanation = one_paragraph_explainer.run(query, context)
    # explanation = ""
    # one_paragraph_explanation = ""

    # converation_context = f"Explanation: {explanation}\n\nOne Paragraph Explanation: {one_paragraph_explanation}"
    # conversation = mcqa_conv.run(query, converation_context)
    conversation = Conversation(turns=[])

    return explanation, one_paragraph_explanation, faiss_rag_docs, bm25_rag_docs, usefulness_triplets_docs, conversation

def thread_func(batch: pd.DataFrame, batch_save_path: Path):
    outputs = []
    
    for _, row in batch.iterrows():
        question = row["question"]
        options = [row["option1"], row["option2"], row["option3"], row["option4"]]
        answer = row["answer"]
        explanation, one_paragraph_explanation, faiss_rag_docs, bm25_rag_docs, usefulness_triplets_docs, conversation = pipeline(question, options, answer)

        entry = {
            'id': row['id'],
            "question": row['question'],
            "option1": row['option1'],
            "option2": row['option2'],
            "option3": row['option3'],
            "option4": row['option4'],
            "answer": row['answer'],
            "faiss_rag_docs": [doc.model_dump()  for doc in faiss_rag_docs],
            "bm25_rag_docs": [doc.model_dump() for doc in bm25_rag_docs],
            "usefulness_triplets_docs": [t.model_dump() for t in usefulness_triplets_docs],
            "explanation": explanation,
            "one_paragraph_explanation": one_paragraph_explanation,
            "conversation": conversation.model_dump()
        }
        outputs.append(entry)
    
    save_df = pd.DataFrame(outputs)
    save_df.to_json(batch_save_path, orient="records", lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument("csv_path", type=Path, help="Path to the csv file")
    parser.add_argument("--random_sample", type=int, default=None, help="Number of random samples to take")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for parallel processing")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    if ('question' not in df.columns or 'option1' not in df.columns or 'option2' not in df.columns or 'option3' not in df.columns or 'option4' not in df.columns or 'answer' not in df.columns):
        raise ValueError("The mcqa csv file does not have 'question', 'option1', 'option2', 'option3', 'option4', 'answer' columns.")
    
    if args.random_sample is not None:
        df = df.sample(n=args.random_sample, random_state=42)
    
    # faiss_triplets_rag = FaissTripletsRag(top_k=10)
    # bm25_triplets_rag = BM25TripletsRag(top_k=10)
    # triplet_usefulness = TripletsUsefulness()
    faiss_rag = FaissRag(top_k=5)
    bm25_rag = BM25Rag(top_k=5)
    triplet_explainer = TripletRAGExplainer()
    one_paragraph_explainer = OneParagraphExplainer() 
    # mcqa_conv = MCQAConversation()

    timestring = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_folder = Path(f"./outputs/{csv_path.stem}/{timestring}")
    run_folder.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        completed_batches = get_completed_batch_numbers(run_folder)
        for i in range(0, len(df), batch_size):
            if i//batch_size in completed_batches:
                print(f"Skipping batch {i//batch_size}")
                continue
            futures.append(executor.submit(thread_func, df[i:i+batch_size], run_folder / f"batch_{i//batch_size}.jsonl"))
        for future in tqdm(futures):
            future.result()
    
    merge_jsonl_files(run_folder)