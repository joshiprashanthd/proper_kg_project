import igraph as ig
import numpy as np
from src.utils import logger

class PageRankPruner:
    def run(self, G: ig.Graph, personalized_nodes: list[str], top_k=20):
        eps = 1e-6
        if len(personalized_nodes) == 0 or G.vcount() == 0:
            return G
        
        try:
            vertex_scores = G.personalized_pagerank(vertices=G.vs, directed=False, reset_vertices=personalized_nodes)
            min_max_normalized = (np.array(vertex_scores) - np.min(vertex_scores)) / (np.max(vertex_scores) - np.min(vertex_scores) + eps)
            G.vs['score'] = min_max_normalized
            selected_vertices = G.vs.select(name_notin=personalized_nodes)
            top_selected_vertices = sorted(list(selected_vertices), key=lambda vertex: vertex['score'], reverse=True)[:top_k]
            top_selected_vertices_names = [vertex['name'] for vertex in top_selected_vertices]
            pruned_G = G.induced_subgraph(top_selected_vertices_names + personalized_nodes)
            return pruned_G
        except Exception as e:
            logger.error(str(e))
            logger.warning("Returning empty graph")
            return ig.Graph()
