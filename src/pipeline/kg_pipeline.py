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

    def run(self, question, answer, qna_context_prefix: str = "", qna_context: str = "", top_k_triplets=20, pruned_top_k_nodes=20):
        logger.info(f"Running pipeline...")

        logger.info(f"Question = {question}") 
        logger.info(f"Answer = {answer}")
        logger.info(f"QnA Context = {qna_context}")

        q_ents, c_ents, a_ents = self.ner.run_mult([question, qna_context, answer])

        q_ents = list(set(flatten([ent.split(" ") for ent in q_ents])))
        c_ents = list(set(flatten([ent.split(" ") for ent in c_ents])))
        a_ents = list(set(flatten([ent.split(" ") for ent in a_ents])))

        logger.info("Retrieving triplets...")
        result_df = self.triplets_retriever.run(q_ents + c_ents + a_ents, top_k=top_k_triplets)
        unique_nodes = list(set(result_df['source'].tolist() + result_df['target'].tolist()))

        logger.info(f"# Triplets: {len(result_df)}")
        logger.info(f"# Unique nodes: {len(unique_nodes)}")
        logger.info(f"Unique nodes: {unique_nodes}")


        logger.info(f"# Question Entities: {len(q_ents)}")
        logger.info(f"Question Entities: {q_ents}")

        logger.info(f"# Option Entities: {len(c_ents)}")
        logger.info(f"Option Entities: {c_ents}")

        logger.info(f"# Answer Entities: {len(a_ents)}")
        logger.info(f"Answer Entities: {a_ents}")

        q_nodes = graph_find_related_nodes_to(unique_nodes, q_ents)
        c_nodes = graph_find_related_nodes_to(unique_nodes, c_ents)
        a_nodes = graph_find_related_nodes_to(unique_nodes, a_ents)

        logger.info(f"Question Nodes: {q_nodes}")
        logger.info(f"Question Nodes: {q_nodes}")

        logger.info(f"Option Nodes: {c_nodes}")
        logger.info(f"Option Nodes: {c_nodes}")

        logger.info(f"Answer Nodes: {a_nodes}")
        logger.info(f"Answer Nodes: {a_nodes}")

        logger.info("Creating subgraphs...")
        q_G = self.subgraph_creator.run(self.G, q_nodes) if len(q_nodes) > 0 else None
        c_G = self.subgraph_creator.run(self.G, c_nodes) if len(c_nodes) > 0 else None
        a_G = self.subgraph_creator.run(self.G, q_nodes, to_nodes=a_nodes) if len(q_nodes) > 0 else (None if len(a_nodes) == 0 else self.subgraph_creator.run(self.G, a_nodes))

        if q_G is not None:
            logger.info(f"Question Graph: Nodes: {q_G.vcount()}, Edges: {q_G.ecount()}")
        
        if c_G is not None:
            logger.info(f"Option Graph Stats: Nodes: {c_G.vcount()}, Edges: {c_G.ecount()}")
        
        if a_G is not None:
            logger.info(f"Answer Graph Stats: Nodes: {a_G.vcount()}, Edges: {a_G.ecount()}")
        
        logger.info("Pruning subgraphs...")
        pruned_q_G = self.pruner.run(q_G, q_nodes, top_k=pruned_top_k_nodes) if q_G is not None else None
        pruned_c_G = self.pruner.run(c_G, c_nodes, top_k=pruned_top_k_nodes) if c_G is not None else None
        pruned_a_G = self.pruner.run(a_G, q_nodes + a_nodes, top_k=pruned_top_k_nodes) if a_G is not None else None

        reasoning_paths = []
        if pruned_q_G is not None and pruned_q_G.vcount() > 0:
            logger.info(f"Pruned Question Graph Stats: Nodes: {pruned_q_G.vcount()}, Edges: {pruned_q_G.ecount()}")
            q_rp = self.reasoning_path_generator.run(pruned_q_G, q_nodes)
            reasoning_paths += q_rp if q_rp is not None else []
        else:
            logger.warning("No pruned question graph")
        
        if pruned_c_G is not None and pruned_c_G.vcount() > 0:
            logger.info(f"Pruned Option Graph Stats: Nodes: {pruned_c_G.vcount()}, Edges: {pruned_c_G.ecount()}")
            op_rp = self.reasoning_path_generator.run(pruned_c_G, c_nodes)
            reasoning_paths += op_rp if op_rp is not None else []
        else:
            logger.warning("No pruned option graph")

        if pruned_a_G is not None and pruned_a_G.vcount() > 0:
            logger.info(f"Pruned Answer Graph Stats: Nodes: {pruned_a_G.vcount()}, Edges: {pruned_a_G.ecount()}")
            ans_rp = self.reasoning_path_generator.run(pruned_a_G, a_nodes)
            reasoning_paths += ans_rp if ans_rp is not None else []
        else:
            logger.warning("No pruned answer graph")

        if len(reasoning_paths) == 0:
            logger.warning("No reasoning paths found")
        else:
            logger.info(f"# Reasoning Paths: {len(reasoning_paths)}")
            logger.info(f"Reasoning Paths: {reasoning_paths}")

        graph_context_prefix = "Reasoning Paths: " if len(reasoning_paths) > 0 else ""
        graph_context = graph_reasoning_paths_to_text(reasoning_paths) if len(reasoning_paths) > 0 else ""
        logger.info(f"Reasoning Paths: {graph_context}")
        
        logger.info("Generating explanation...")
        explanation = self.explainer.run(question, answer, qna_context_prefix, qna_context, graph_context_prefix, graph_context)
        return explanation, graph_context