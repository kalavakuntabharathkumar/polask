from __future__ import annotations
import os
from .models import Answer, AskRequest, Citation
from .retrieval import PolicyRetriever

SYSTEM_RULES = (
    "Answer only from the supplied policy excerpts. If the excerpts do not support an answer, "
    "say that you cannot determine it from the available policy documents. Never invent policy."
)


def answer_question(request: AskRequest, retriever: PolicyRetriever) -> Answer:
    hits = retriever.search(request.question, request.top_k)
    if not hits:
        return Answer(
            answer="I couldn't find a relevant policy excerpt. Please contact HR or search using different terms.",
            citations=[],
            confidence="low",
            mode="demo",
        )

    # Safe deterministic demo mode. It deliberately does not pretend to call an LLM.
    excerpts = [hit.document.text.strip() for hit in hits]
    citations = [
        Citation(
            document_id=hit.document.id,
            title=hit.document.title,
            excerpt=hit.document.text[:260].strip(),
        )
        for hit in hits
    ]
    answer = (
        "Relevant policy material was found. Review the cited excerpts below before acting: "
        + " ".join(excerpts[:2])
    )
    confidence = "high" if hits[0].score >= 2.0 else "medium"
    return Answer(answer=answer, citations=citations, confidence=confidence, mode="demo")


def provider_configuration() -> dict[str, bool]:
    """Reports whether provider credentials are present without exposing secrets."""
    return {
        "azure_openai_configured": bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY")),
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
    }
