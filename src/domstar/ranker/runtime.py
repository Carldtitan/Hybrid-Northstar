"""Inference helpers for the DOM candidate ranker."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from domstar.dom.schema import DOMCandidate


@dataclass(slots=True)
class RankedCandidate:
    """One candidate along with its ranker score."""

    candidate: DOMCandidate
    score: float


class DOMRanker:
    """Thin wrapper around a sequence-classification cross-encoder."""

    def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, candidates: list[DOMCandidate], batch_size: int = 32) -> list[RankedCandidate]:
        """Rank candidates by task relevance."""

        ranked: list[RankedCandidate] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            tokenized = self.tokenizer(
                [query] * len(batch),
                [candidate.to_ranker_text() for candidate in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**tokenized).logits
            if logits.ndim == 1 or logits.shape[-1] == 1:
                scores = torch.sigmoid(logits.squeeze(-1)).tolist()
            else:
                scores = torch.softmax(logits, dim=-1)[:, 1].tolist()
            ranked.extend(
                RankedCandidate(candidate=candidate, score=float(score))
                for candidate, score in zip(batch, scores, strict=True)
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked
