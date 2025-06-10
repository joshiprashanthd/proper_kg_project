import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from src.components.retrieval.subgraph.main_subgraph_retrieval import MainSubgraphRetrieval
from dotenv import load_dotenv
load_dotenv()


ret = MainSubgraphRetrieval()
subgraph, foundNodes = ret.run(
    entities=["depression", "anxiety", "fever"],
)

print(f"Subgraph nodes = {subgraph.vcount()} , edges = {subgraph.ecount()}")
print("FOUND NODES = ", foundNodes)