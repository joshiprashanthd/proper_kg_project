from pydantic import BaseModel, Field
from typing import Literal

class SymptomPhrase(BaseModel):
    phrase: str = Field(description="The symptom phrase.")
    symptom: str = Field(description="The symptom of the symptom phrase.")
    analysis: str = Field(description="The analysis of the symptom phrase.")

class TripletLabel(BaseModel):
    triplet: str = Field(description="triplet found in the prompt itself")
    justification: str = Field(description="Whether it is relevant to the Question and Answer, and whether it provides insights that could help in arriving at the Answer, with a 4 to 5 lines of justification.")
    label: Literal["Useful", "Not Useful"] = Field(description="usefulness label of the triplet")
    score: int = Field(description="usefulness score of the triplet between 1 and 5")