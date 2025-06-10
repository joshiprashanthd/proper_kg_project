from .base_subgraph_creator import BaseSubgraphCreator
import igraph as ig
from more_itertools import flatten

class FirstShortestPathSubgraphCreator(BaseSubgraphCreator):
    def run(self, ref_G: ig.Graph, nodes: list[int], to_nodes: list[int] = None) -> ig.Graph:
        # assert all([isinstance(node, int) for node in nodes]), "Nodes should be a list of integers"
        # assert to_nodes is None or all([isinstance(node, int) for node in to_nodes]), "To nodes should be a list of integers"
        if len(nodes) > 0 and isinstance(nodes[0], str):
            nodes = [ref_G.vs.find(name=node).index for node in nodes]
        if to_nodes is not None and len(to_nodes) > 0 and isinstance(to_nodes[0], str):
            to_nodes = [ref_G.vs.find(name=node).index for node in to_nodes]

        all_vertices = set(nodes)
        for i, start_node in enumerate(nodes):
            other = to_nodes if to_nodes is not None else nodes[i+1:]
            paths = ref_G.get_shortest_paths(start_node, other, mode='all')
            all_vertices.update(list(flatten(paths)))
        return ref_G.induced_subgraph(list(all_vertices))