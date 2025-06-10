from .base_subgraph_retrieval import BaseSubgraphRetrieval
import igraph as ig
from src.utils import logger

class MainSubgraphRetrieval(BaseSubgraphRetrieval):
    def run(self, entities: list[str]):
        cypher_query = """
        WITH $entities as entities

        MATCH (t:Term)
        WHERE lower(t.name) IN [e IN entities | lower(e)]
        WITH COLLECT(DISTINCT t) AS originalTerms

        UNWIND range(0, size(originalTerms) - 2) AS i
        UNWIND range(i+1, size(originalTerms) - 1) AS j
        WITH originalTerms[i] AS headTerm, originalTerms[j] AS tailTerm, originalTerms

        MATCH p = ALL SHORTEST (headTerm)<--(:Concept|Code) ((:Concept|Code)-->(:Concept|Code))+ (:Concept|Code)-->(tailTerm)
        WITH nodes(p) AS pathNodes, relationships(p) AS rels, originalTerms

        UNWIND range(1, size(pathNodes)-3) AS i
        WITH pathNodes[i] AS headConcept, pathNodes[i+1] AS tailConcept, rels[i] AS r, originalTerms

        MATCH 
            (headTerm:Term)<--(headConcept)-->(headDef:Definition),
            (headConcept)-->(headSemantic:Semantic),
            (headConcept)-[r]->(tailConcept),
            (tailTerm:Term)<--(tailConcept)-->(tailDef:Definition),
            (tailConcept)-->(tailSemantic:Semantic)
            
        WITH COLLECT(DISTINCT headDef.DEF) AS headDefs, COLLECT(DISTINCT tailDef.DEF) AS tailDefs, headTerm, tailTerm, headSemantic, tailSemantic, r, originalTerms

        SET headTerm.definitions = headDefs
        SET headTerm.semantic_type = headSemantic.name
        SET tailTerm.definitions = tailDefs
        SET tailTerm.semantic_type = tailSemantic.name

        MERGE (headTerm)-[:$(type(r))]->(tailTerm)

        RETURN headTerm, tailTerm, type(r) AS relationType, originalTerms
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