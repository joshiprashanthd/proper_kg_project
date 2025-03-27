import pytest
from src.pipeline.kg_pipeline import KGPipeline
from src.components.ner.spacy_ner import SpacyNER
from src.components.subgraph_retrieval.related_nodes_retriever import RelatedNodesRetriever
from src.components.subgraph_pruning.pagerank_pruner import PageRankPruner
from src.components.reasoning_paths.path_finder import PathFinder
from src.components.explanation.llm_explainer import LLMExplainer

def test_kg_pipeline_initialization():
    pipeline = KGPipeline()
    assert pipeline is not None

def test_ner_component():
    ner = SpacyNER()
    text = "Apple is looking at buying U.K. startup for $1 billion"
    entities = ner.recognize_entities(text)
    assert isinstance(entities, list)

def test_subgraph_retrieval():
    retriever = RelatedNodesRetriever()
    entities = ["Apple", "U.K."]
    related_nodes = retriever.retrieve_related_nodes(entities)
    assert isinstance(related_nodes, list)

def test_subgraph_pruning():
    pruner = PageRankPruner()
    subgraph = {"nodes": ["A", "B", "C"], "edges": [("A", "B"), ("B", "C")]}
    pruned_subgraph = pruner.prune(subgraph)
    assert isinstance(pruned_subgraph, dict)

def test_reasoning_path_generation():
    path_finder = PathFinder()
    pruned_subgraph = {"nodes": ["A", "B"], "edges": [("A", "B")]}
    paths = path_finder.generate_paths(pruned_subgraph)
    assert isinstance(paths, list)

def test_explanation_generation():
    explainer = LLMExplainer()
    paths = [["A", "B"], ["B", "C"]]
    explanations = explainer.generate_explanations(paths)
    assert isinstance(explanations, list)