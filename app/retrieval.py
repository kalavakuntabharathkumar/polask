from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from .models import PolicyDocument

TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass
class SearchHit:
    document: PolicyDocument
    score: float


class PolicyRetriever:
    """Small local lexical retriever; replace with Azure AI Search for deployment."""

    def __init__(self, documents: Iterable[PolicyDocument]):
        self.documents = list(documents)
        self.term_counts = [Counter(tokenize(doc.title + " " + doc.text)) for doc in self.documents]
        self.doc_freq: Counter[str] = Counter()
        for counts in self.term_counts:
            self.doc_freq.update(counts.keys())
        self.avg_len = sum(sum(c.values()) for c in self.term_counts) / max(len(self.term_counts), 1)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PolicyRetriever":
        import json
        documents = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                documents.append(PolicyDocument.model_validate(json.loads(line)))
        return cls(documents)

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        terms = tokenize(query)
        if not terms:
            return []
        n = len(self.documents)
        hits: list[SearchHit] = []
        for doc, counts in zip(self.documents, self.term_counts):
            length = sum(counts.values())
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = 1.0 + __import__("math").log((n + 1) / (self.doc_freq[term] + 1))
                denom = tf + 1.2 * (0.25 + 0.75 * length / max(self.avg_len, 1))
                score += idf * (tf * 2.2) / denom
            if score > 0:
                hits.append(SearchHit(doc, score))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
