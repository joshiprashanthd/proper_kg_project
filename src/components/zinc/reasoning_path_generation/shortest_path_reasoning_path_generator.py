from src.components.reasoning_path_generation import BaseReasoningPathGenerator
import igraph as ig
from src.utils import logger

class ShortestPathReasoningPathGenerator(BaseReasoningPathGenerator):
    def __init__(self):
        super().__init__()

    def run(self, G: ig.Graph, fixed_nodes: list[str]):
        logger.info(f"Generating reasoning paths for nodes: {fixed_nodes}")
        reasoning_paths = []
        for node in fixed_nodes:
            to = G.vs.select(name_notin=[node])
            paths = G.get_all_shortest_paths(node, to, mode='in')
            for path in paths:
                if len(path) <= 0: continue
                path = list(reversed(path))
                new_path = []
                for i in range(len(path) - 1):
                    src = G.vs[path[i]]
                    trg = G.vs[path[i+1]]
                    edge = G.es.find(_source=src.index, _target=trg.index)
                    new_path += [src['name'], edge['type']]
                new_path += [G.vs[path[-1]]['name']]
                reasoning_paths.append(new_path)
        return reasoning_paths