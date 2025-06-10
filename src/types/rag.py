from pydantic import BaseModel, Field

class RagDocument(BaseModel):
    content: str = Field(description="The content of the document")
    score: float = Field(description="The score of the document", default=0.0)
    source: str = Field(description="The source of the document")
    type: str = Field(description="The type of the document")

    def __hash__(self):
        return hash((self.content, self.score, self.source, self.type))

    def __eq__(self, other):
        return self.content == other.content and self.score == other.score and self.source == other.source and self.type == other.type
        