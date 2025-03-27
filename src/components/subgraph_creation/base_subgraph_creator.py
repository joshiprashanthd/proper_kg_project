import igraph as ig

class BaseSubgraphCreator:
    def __init__(self):
        pass

    def run(self, ref_G: ig.Graph, nodes: list[int], to_nodes: list[int] = None) -> ig.Graph:
        pass
        