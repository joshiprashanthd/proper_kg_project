import sys
sys.path.append("/home/sracha/proper_kg_project")

from numpy.strings import str_len
import pandas as pd
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datetime import datetime


from src.components.usefulness import TripletsUsefulness
from src.components.faiss_rag import FaissRag
from src.components.bm25_rag import BM25Rag
from src.components.faiss_triplets_rag import FaissTripletsRag
from src.components.bm25_triplets_rag import BM25TripletsRag
from src.components.symptom_phrases import SymptomPhrasesExtractor
from src.components.explanation import OneParagraphExplainer, TripletRAGExplainer
from src.utils import get_completed_batch_numbers, keep_first_definition, merge_jsonl_files
from src.types import TripletLabel, RagDocument

def build_naive_rag_query(docs: list[TripletLabel]):
    DOC_TEMPLATE = """{triplet}
{justification}
{label}
"""
    return "\n".join([DOC_TEMPLATE.format(triplet=doc.triplet, justification=doc.justification, label=doc.label) for doc in docs])

def build_explanation_context(rag_docs: list[RagDocument], triplet_docs: list[TripletLabel]):
    CONTEXT_TEMPLATE = """Below is the additional context for query:
<rag_paragraphs>
{rag_results}
</rag_paragraphs>

<triplets>
{triplets}
</triplets>
"""
    
    context = CONTEXT_TEMPLATE.format(
        rag_results="\n\n".join([doc.content for doc in rag_docs]),
        triplets="\n\n".join(keep_first_definition([t.triplet for t in triplet_docs]))
    )

    return context

def pipeline(phrase: str, symptom: str, analysis: str):
    query = f"{phrase}\n{symptom}\n{analysis}"
    
    faiss_triplets_rag_docs = faiss_triplets_rag.run(query)
    bm25_triplets_rag_docs = bm25_triplets_rag.run(query)
    
    usefulness_triplets_docs = triplet_usefulness.run(query, keep_first_definition([t.content for t in faiss_triplets_rag_docs + bm25_triplets_rag_docs]))
    usefulness_triplets_docs = [t for t in sorted(usefulness_triplets_docs, key=lambda t: t.score, reverse=True) if t.label == "Useful"]
    
    naive_rag_query = f"{query}\n{build_naive_rag_query(usefulness_triplets_docs)}"
    
    faiss_rag_docs = faiss_rag.run(naive_rag_query) 
    bm25_rag_docs = bm25_rag.run(naive_rag_query)

    context = build_explanation_context(list(set(faiss_rag_docs + bm25_rag_docs)), usefulness_triplets_docs)
    
    explanation = triplet_rag_explainer.run(query, context)
    one_paragraph_explanation = one_paragraph_explainer.run(query, context)
    
    return explanation, one_paragraph_explanation, faiss_rag_docs, bm25_rag_docs, usefulness_triplets_docs

def thread_func(batch: pd.DataFrame, batch_save_path: Path):
    outputs = {
        "text": [],
        "label": [],
        "phrase": [],
        "symptom": [],
        "analysis": [],
        "explanation": [],
        "one_paragraph_explanation": [],
        "faiss_rag_docs": [],
        "bm25_rag_docs": [],
        "usefulness_triplets_docs": [],
    }

    for _, row in batch.iterrows():
        text = row["text"]
        label = row["label"]

        symptom_phrases = symptom_phrase_extractor.run(text)

        for sp in symptom_phrases:
            phrase = sp.phrase
            symptom = sp.symptom
            analysis = sp.analysis

            explanation, one_paragraph_explanation, faiss_rag_docs, bm25_rag_docs, usefulness_triplets_docs = pipeline(phrase, symptom, analysis)

            outputs['text'].append(text)
            outputs['label'].append(label)
            outputs['phrase'].append(phrase)
            outputs['symptom'].append(symptom)
            outputs['analysis'].append(analysis)
            outputs['explanation'].append(explanation)
            outputs['one_paragraph_explanation'].append(one_paragraph_explanation)
            outputs['faiss_rag_docs'].append([doc.model_dump()  for doc in faiss_rag_docs])
            outputs['bm25_rag_docs'].append([doc.model_dump() for doc in bm25_rag_docs])
            outputs['usefulness_triplets_docs'].append([t.model_dump() for t in usefulness_triplets_docs])
    
    save_df = pd.DataFrame(outputs)
    save_df.to_json(batch_save_path, orient="records", lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline")
    parser.add_argument("csv_path", type=str, help="Path to the csv file")
    parser.add_argument("--random_sample", type=int, default=None, help="Number of random samples to take")
    parser.add_argument("--workers", type=int, default=32, help="Number of workers for parallel processing")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)
    
    if "text" not in df.columns or 'label' not in df.columns:
        raise ValueError("The csv file does not have 'text' or 'label' column.")
    
    if args.random_sample is not None:
        df = df.sample(n=args.random_sample, random_state=42)
    
    symptom_phrase_extractor = SymptomPhrasesExtractor()
    faiss_triplets_rag = FaissTripletsRag(top_k=5)
    bm25_triplets_rag = BM25TripletsRag(top_k=5)
    triplet_usefulness = TripletsUsefulness()
    faiss_rag = FaissRag(top_k=3)
    bm25_rag = BM25Rag(top_k=3)
    triplet_rag_explainer = TripletRAGExplainer()
    one_paragraph_explainer = OneParagraphExplainer()

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