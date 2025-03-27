import igraph as ig

class BaseReasoningPathGenerator:
    def __init__(self):
        pass

    def run(self, graph: ig.Graph, fixed_nodes: list[str]):
        raise NotImplementedError("Subclasses should implement this method.")