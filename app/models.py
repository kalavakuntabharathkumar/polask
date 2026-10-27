from pydantic import BaseModel, Field
from typing import Literal


class PolicyDocument(BaseModel):
    id: str
    title: str
    text: str
    category: str = "general"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)


class Citation(BaseModel):
    document_id: str
    title: str
    excerpt: str


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["low", "medium", "high"]
    mode: Literal["demo", "azure_openai"] = "demo"
