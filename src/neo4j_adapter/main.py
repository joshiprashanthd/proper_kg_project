from neo4j import GraphDatabase
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import os

load_dotenv()

class Neo4jAdapter:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]