from src.pipeline.kg_pipeline import KGPipeline
from src.components.ner import OpenAINER
from src.components.retrieval.triplets import BM25TripletsRetriever
from src.components.subgraph_creation import FirstShortestPathSubgraphCreator, ConstrainedShortestPathSubgraphCreator
from src.components.pruning import PageRankPruner
from src.components.reasoning_path_generation import ShortestPathReasoningPathGenerator
from src.components.explanation import MainExplainer
from src.utils import load_graph
from logging import info

G, nodes_df, edges_df = load_graph("../data/primekg", return_df=True)

ner = OpenAINER()


info("Loading Retriever...")
triplets_retriever = BM25TripletsRetriever(edges_df)
info("Retriever Loaded")

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

question = "What disorder is characterized by an excessive accumulation of iron in the body?"
answer = "Hemochromatosis"
options = ["Hemochromatosis", "Thalassemia", "Sickle cell anemia", "Iron deficiency anemia"]

question, explanation, answer = pipeline.run(question, 
             answer, 
             qna_context_prefix="Options:", 
             qna_context="\n".join(options),
             top_k_triplets=10,
             pruned_top_k_nodes=10)

print("Question:", question)
print("Explanation:", explanation)
print("Answer:", answer)