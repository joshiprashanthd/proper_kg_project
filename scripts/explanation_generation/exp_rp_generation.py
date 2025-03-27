import csv
import pandas as pd
from pathlib import Path

from src.pipeline.kg_pipeline import KGPipeline, logger
from src.components.ner import OpenAINER
from src.components.retrieval.triplets import BM25TripletsRetriever
from src.components.subgraph_creation import FirstShortestPathSubgraphCreator, ConstrainedShortestPathSubgraphCreator
from src.components.pruning import PageRankPruner
from src.components.reasoning_path_generation import ShortestPathReasoningPathGenerator
from src.components.explanation import MainExplainer
from src.utils import load_graph, logger
from tqdm import tqdm

# disable logging to stdout
logger.removeHandler(logger.handlers[0])


csv_paths = {}
for file in Path("/home/sracha/proper_kg_project/data/qna/combined/preprocessed").iterdir():
    prefix = file.stem.split("_")[0]
    if "train" in file.stem:
        csv_paths[prefix] = Path(file)

G, nodes_df, edges_df = load_graph("/home/sracha/proper_kg_project/data/primekg", return_df=True, remove_node_types=['gene/protein'])

logger.info(f"Vertices Count = {G.vcount()}, Edges Count = {G.ecount()}")

ner = OpenAINER()
logger.info("Loading Retriever...")
triplets_retriever = BM25TripletsRetriever(edges_df)
logger.info("Retriever Loaded")
# subgraph_creator = ConstrainedShortestPathSubgraphCreator()
subgraph_creator = FirstShortestPathSubgraphCreator()
pruner = PageRankPruner()
reasoning_path_generator = ShortestPathReasoningPathGenerator()
explainer = MainExplainer()

pipeline = KGPipeline(
    G=G,
    ner=ner,
    triplets_retriever=triplets_retriever,
    subgraph_creator=subgraph_creator,
    pruner=pruner,
    reasoning_path_generator=reasoning_path_generator,
    explainer=explainer
)


except_for = ['medquad', 'bioasq', 'pubmed']

for name, path in tqdm(csv_paths.items()):
    logger.info(f"Running for {name}")

    if name in except_for:
        continue

    csv_writer = csv.writer(open(name + "_100_exp_rp.csv", "w"))
    df = pd.read_csv(path)
    df = df.sample(min(100, len(df)), random_state=42)

    logger.info(f"Total Rows of {name}: {len(df)}")

    csv_writer.writerow(df.columns.tolist() + ['explanation', 'reasoning_paths'])

    for i, row in tqdm(df.iterrows(), desc=f"Running {name}", total=len(df)):
        question = row['question']
        answer = row['answer']

        options = []
        if 'option1' in row.keys():
            options = [row[f"option{i}"] for i in range(1, 5)]

        explanation, reasoning_paths = pipeline.run(question, 
                    answer, 
                    qna_context_prefix="Options:" if len(options) > 0 else "", 
                    qna_context="\n".join(options) if len(options) > 0 else "",
                    top_k_triplets=20,
                    pruned_top_k_nodes=20)

        csv_writer.writerow(row.values.tolist() + [explanation, reasoning_paths])