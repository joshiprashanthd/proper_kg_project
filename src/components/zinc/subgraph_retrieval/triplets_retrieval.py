from .base_subgraph_retrieval import BaseSubgraphRetrieval
import igraph as ig
from src.utils import logger

class TripletsRetrieval(BaseSubgraphRetrieval):
    def run(self, entities: list[str]):
        cypher_query = """
        WITH $entities as entities

        MATCH (t:Term)
        WHERE lower(t.name) IN [e IN entities | lower(e)]
        WITH COLLECT(DISTINCT t) AS originalTerms

        
        """
        
        logger.info(f"Running neo4j run_query on entities {entities}")
        result =  self.neo4j_adapter.run_query(cypher_query, {"entities": entities})

        graph = ig.Graph(directed=True)

        originalTerms = None

        for record in result:
            head_term = record['headTerm']
            tail_term = record['tailTerm']
            relationType = record['relationType']
            originalTerms = record['originalTerms']

            if ('name' not in graph.vs) or (head_term['name'] not in graph.vs['name']):
                graph.add_vertex(name=head_term['name'], sui=head_term['SUI'], type=head_term['semantic_type'], definitions=head_term['definitions'])
            
            if ('name' not in graph.vs) or (tail_term['name'] not in graph.vs['name']):
                graph.add_vertex(name=tail_term['name'], sui=tail_term['SUI'], type=tail_term['semantic_type'], definitions=tail_term['definitions'])

            graph.add_edge(head_term['name'], tail_term['name'], type=relationType)


        if originalTerms is None:
            logger.warning("No original terms found in the result.")
            return None, []

        foundNodes = [record['name'] for record in originalTerms if len(graph.vs.select(name = record['name'])) > 0]
        
        logger.info(f"Retrieved Subgraph with {graph.vcount()} nodes and {graph.ecount()} edges.")
        logger.info(f"Found nodes: {foundNodes}")
        
        return graph, foundNodes