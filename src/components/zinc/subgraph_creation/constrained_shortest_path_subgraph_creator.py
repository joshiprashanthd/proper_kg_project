from .base_subgraph_creator import BaseSubgraphCreator
import igraph as ig
from collections import deque
from logging import info

class ConstrainedShortestPathSubgraphCreator(BaseSubgraphCreator):
    def __init__(self, min_path_length=2, max_path_length=4):
        super().__init__()
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length

    def run(self, ref_G: ig.Graph, nodes, to_nodes = None):
        # assert all([isinstance(node, int) for node in nodes]), "Nodes should be a list of integers"
        # assert to_nodes is None or all([isinstance(node, int) for node in to_nodes]), "To nodes should be a list of integers"

        if len(nodes) > 0 and isinstance(nodes[0], str):
            nodes = [ref_G.vs.find(name=node).index for node in nodes]
        if to_nodes is not None and len(to_nodes) > 0 and isinstance(to_nodes[0], str):
            to_nodes = [ref_G.vs.find(name=node).index for node in to_nodes]

        info(f"nodes: {nodes}")
        info(f"to_nodes: {to_nodes}")

        vertices = set()

        for i, src in enumerate(nodes):
            queue = deque([(src, [src])])  # (current_node, path_so_far)
            visited = set()
            other_nodes = to_nodes if to_nodes is not None else nodes[i+1:]
            
            while len(queue) > 0:
                node, path = queue.popleft()
                path_length = len(path) - 1
                if path_length > self.max_path_length:
                    continue
                if path_length >= self.min_path_length and node in other_nodes:
                    vertices.update(path)
                visited.add(node)
                for neighbor in ref_G.neighbors(node, mode='out'):
                    if neighbor not in path:
                        queue.append((neighbor, path + [neighbor]))

        return ref_G.induced_subgraph(list(vertices))

    