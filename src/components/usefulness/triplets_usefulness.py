from pydantic import BaseModel
from src.utils import OpenAIModel, logger
from src.types import TripletLabel

class ResponseFormat(BaseModel):
    triplets_with_labels: list[TripletLabel]

class TripletsUsefulness:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = OpenAIModel(model_name, 'cuda')
        self.prompt_template = """You are medical expert and mental health expert. Your goal is to verify whether concepts or entities in the triplet are related to the provided query.

Each triplet is structured in this format:
(Head: <head node name>, Definition: <head node definition>)-[<relation name>]->(Tail: <tail node name>, Definition: <tail node definition>)


Task Instructions:
- For each triplet, determine if the "Head" node, "Tail" node, and the "Relation" between them are directly relevant to the core concepts, entities, conditions, treatments, or symptoms discussed in the provided query.
- Are the terms in the triplet (Head, Tail, Relation) explicitly mentioned or strongly implied in the query?
- Do the definitions provided for the Head and Tail nodes align with the context of the query?
- Is the relationship described in the triplet logically connected to the information sought or provided in the query?
- If a triplet contains a term that is a broader category or a related but not directly discussed entity, does its inclusion still contribute to understanding the context of the query?
- For each triplet, evaluate whether it offers valuable insights, logical connections, or supporting information that could be used to deduce, explain, or arrive at the Answer given the query.
- Does the triplet explain a causal link, a risk factor, a symptom-condition relationship, a treatment-outcome relationship, or a definitional aspect that is crucial for the Answer?
- Does the triplet provide evidence or reasoning that directly supports the correctness of the Answer?
- Could this triplet serve as a stepping stone in a logical chain of reasoning from the query to the Answer?
- If a triplet is relevant but provides redundant information already explicitly stated or easily inferable from the query, does it still offer unique insight for generating the answer? Clearly differentiate between relevance and unique insight.

IMPORTANT: Once you find which triplets are useful, rerank them based on their usefulness and give them a score between 1 to 5, 1 being the least useful and 5 being the most useful.

Query: {query}
Triplets:
{triplets}
"""
    def run(self, query: str, triplets: list[str]) -> list[TripletLabel]:
        response = self.model.generate_text(
            self.prompt_template.format(query=query, triplets="\n".join(triplets)),
            structured_format=ResponseFormat,
            max_tokens=8192
        )
        logger.info(f"Found {len([triplet for triplet in response.triplets_with_labels if triplet.label == 'Useful'])} useful triplets for query: {query}")
        return response.triplets_with_labels