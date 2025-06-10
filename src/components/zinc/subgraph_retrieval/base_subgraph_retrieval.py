from src.neo4j_adapter import Neo4jAdapter

class BaseSubgraphRetrieval:
    def __init__(self):
        self.neo4j_adapter = Neo4jAdapter()

    def run(self, question_entities: list[str], answer_entities: list[str], context_entities: list[str]):
        """
        Run the subgraph retrieval process.

        Args:
            question_entities (list[str]): List of entities related to the question.
            answer_entities (list[str]): List of entities related to the answer.
            context_entities (list[str]): List of entities related to the context.

        Returns:
            list[dict]: A list of dictionaries containing the retrieved subgraph data.
        """
        raise NotImplementedError("Subclasses should implement this method.")