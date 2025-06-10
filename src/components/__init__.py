from pydantic import BaseModel, Field
from typing import Literal

class RagDocument(BaseModel):
    content: str = Field(description="The content of the document")
    score: float = Field(description="The score of the document", default=0.0)
    source: str = Field(description="The source of the document")

    def __hash__(self):
        return hash((self.content, self.score, self.source))

    def __eq__(self, other):
        return self.content == other.content and self.score == other.score and self.source == other.source