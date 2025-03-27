import igraph as ig

class BasePruner:
    def __init__(self):
        pass

    def run(self, graph: ig.Graph, priority_entities: list[str], top_k: int =20):
        raise NotImplementedError("Subclasses should implement this method.")