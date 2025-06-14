from pydantic import BaseModel, Field
from typing import Literal

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(description="The role of the speaker.")
    content: str = Field(description="The content of the speaker.")
    characteristic: Literal[
        "Symptoms Identification",
        "Disease/Disorder Identification",
        "Follow-up Questions",
        "Suggestions/Recommendations/Solutions",
        "Explanations"
    ] = Field(description="The characteristic of the turn.")

class Conversation(BaseModel):
    turns: list[ConversationTurn] = Field(description="The conversation between user and assistant.")

class MultipleConversations(BaseModel):
    conversations: list[Conversation] = Field(description="The conversations between user and assistant")

class SymptomPhrase(BaseModel):
    phrase: str = Field(description="The symptom phrase.")
    symptom: str = Field(description="The symptom of the symptom phrase.")
    analysis: str = Field(description="The analysis of the symptom phrase.")

    def __hash__(self):
        return hash((self.phrase, self.symptom, self.analysis))
    
    def __eq__(self, other):
        return self.phrase == other.phrase and self.symptom == other.symptom and self.analysis == other.analysis

class SymptomPhraseLabel(BaseModel):
    phrase: str = Field(description="The symptom phrase.")
    label: Literal["Anxiety", "Depression", "Both Anxiety and Depression"] = Field(description="The label of the symptom phrase.")
    justification: str = Field(description="The justification for the label of the symptom phrase.")

    def __hash__(self):
        return hash((self.phrase, self.label, self.justification))
    
    def __eq__(self, other):
        return self.phrase == other.phrase and self.label == other.label and self.justification == other.justification

class TripletLabel(BaseModel):
    triplet: str = Field(description="triplet found in the prompt itself")
    justification: str = Field(description="Whether it is relevant to the Question and Answer, and whether it provides insights that could help in arriving at the Answer, with a 4 to 5 lines of justification.")
    label: Literal["Useful", "Not Useful"] = Field(description="usefulness label of the triplet")
    score: int = Field(description="usefulness score of the triplet between 1 and 5")

    def __hash__(self):
        return hash((self.triplet, self.justification, self.label, self.score))
    
    def __eq__(self, other):
        return self.triplet == other.triplet and self.justification == other.justification and self.label == other.label and self.score == other.score
        