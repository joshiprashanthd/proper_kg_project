from more_itertools import flatten
import igraph as ig

from src.components.ner import BaseNER
from src.components.explanation.base_explainer import BaseExplainer
from src.components.reasoning_path_generation import BaseReasoningPathGenerator
from src.components.pruning import BasePruner
from src.components.retrieval.triplets import BaseTripletsRetriever
from src.components.subgraph_creation import BaseSubgraphCreator
from src.utils import graph_reasoning_paths_to_text, graph_find_related_nodes_to, logger


class KGPipeline:
    def __init__(self,
                 G: ig.Graph,
                 ner: BaseNER, 
                 triplets_retriever: BaseTripletsRetriever,
                 subgraph_creator: BaseSubgraphCreator,
                 pruner: BasePruner,
                 reasoning_path_generator: BaseReasoningPathGenerator,
                 explainer: BaseExplainer):
        
        self.G = G
        self.ner = ner
        self.triplets_retriever = triplets_retriever
        self.subgraph_creator = subgraph_creator
        self.pruner = pruner
        self.reasoning_path_generator = reasoning_path_generator
        self.explainer = explainer

    def run(self, question, answer, qna_context_prefix="", qna_context="", top_k_triplets=20, pruned_top_k_nodes=20):
        logger.info("Running pipeline...")
        logger.info(f"Question = {question}")
        logger.info(f"Answer = {answer}")
        logger.info(f"QnA Context = {qna_context}")

        # Extract entities
        q_ents, c_ents, a_ents = self.ner.run_mult([question, qna_context, answer])
        
        # Process entities
        q_ents = list(set(flatten([ent.split(" ") for ent in q_ents])))
        c_ents = list(set(flatten([ent.split(" ") for ent in c_ents])))
        a_ents = list(set(flatten([ent.split(" ") for ent in a_ents])))
        
        # Retrieve triplets
        logger.info("Retrieving triplets...")
        all_entities = q_ents + c_ents + a_ents
        result_df = self.triplets_retriever.run(all_entities, top_k=top_k_triplets)
        unique_nodes = list(set(result_df['source'].tolist() + result_df['target'].tolist()))

        logger.info(f"# Triplets: {len(result_df)}")
        logger.info(f"# Unique nodes: {len(unique_nodes)}")
        logger.info(f"Unique nodes: {unique_nodes}")
        
        # Log entity information
        for ent_type, ents in [("Question", q_ents), ("Option", c_ents), ("Answer", a_ents)]:
            logger.info(f"# {ent_type} Entities: {len(ents)}")
            logger.info(f"{ent_type} Entities: {ents}")

        # Find related nodes
        q_nodes = graph_find_related_nodes_to(unique_nodes, q_ents)
        c_nodes = graph_find_related_nodes_to(unique_nodes, c_ents)
        a_nodes = graph_find_related_nodes_to(unique_nodes, a_ents)
        
        # Log node information
        for node_type, nodes in [("Question", q_nodes), ("Option", c_nodes), ("Answer", a_nodes)]:
            logger.info(f"{node_type} Nodes: {nodes}")
        
        # Create subgraphs
        logger.info("Creating subgraphs...")
        q_G = self.subgraph_creator.run(self.G, q_nodes) if q_nodes else None
        c_G = self.subgraph_creator.run(self.G, c_nodes) if c_nodes else None
        
        if q_nodes and a_nodes:
            a_G = self.subgraph_creator.run(self.G, q_nodes, to_nodes=a_nodes)
        elif a_nodes:
            a_G = self.subgraph_creator.run(self.G, a_nodes)
        else:
            a_G = None
        
        # Log subgraph stats
        for graph_type, graph in [("Question", q_G), ("Option", c_G), ("Answer", a_G)]:
            if graph is not None:
                logger.info(f"{graph_type} Graph: Nodes: {graph.vcount()}, Edges: {graph.ecount()}")
        
        # Prune subgraphs
        logger.info("Pruning subgraphs...")
        pruned_q_G = self.pruner.run(q_G, q_nodes, top_k=pruned_top_k_nodes) if q_G is not None else None
        pruned_c_G = self.pruner.run(c_G, c_nodes, top_k=pruned_top_k_nodes) if c_G is not None else None
        pruned_a_G = self.pruner.run(a_G, q_nodes + a_nodes, top_k=pruned_top_k_nodes) if a_G is not None else None
        
        # Generate reasoning paths
        reasoning_paths = []
        
        # Process each pruned graph
        for graph_type, graph, nodes in [
            ("Question", pruned_q_G, q_nodes),
            ("Option", pruned_c_G, c_nodes),
            ("Answer", pruned_a_G, a_nodes)
        ]:
            if graph is not None and graph.vcount() > 0:
                logger.info(f"Pruned {graph_type} Graph Stats: Nodes: {graph.vcount()}, Edges: {graph.ecount()}")
                paths = self.reasoning_path_generator.run(graph, nodes)
                if paths:
                    reasoning_paths.extend(paths)
            else:
                logger.warning(f"No pruned {graph_type.lower()} graph")
        
        # Check for reasoning paths
        if not reasoning_paths:
            logger.warning("No reasoning paths found")
            graph_context_prefix = ""
            graph_context = ""
        else:
            logger.info(f"# Reasoning Paths: {len(reasoning_paths)}")
            logger.info(f"Reasoning Paths: {reasoning_paths}")
            graph_context_prefix = "Reasoning Paths: "
            graph_context = graph_reasoning_paths_to_text(reasoning_paths)
            
        logger.info(f"Reasoning Paths: {graph_context}")
        
        # Generate explanation
        logger.info("Generating explanation...")
        explanation = self.explainer.run(
            question,
            answer, 
            qna_context_prefix, 
            qna_context, 
            graph_context_prefix, 
            graph_context
        )
        
        return explanation, graph_context