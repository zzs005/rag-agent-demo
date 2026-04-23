from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.retrieval.schemas import RetrievedChunk


@dataclass
class RefusalDecision:
    should_refuse: bool
    reason: Optional[str]
    top1_score: float
    top2_score: float
    avg_top3_score: float


class RuleBasedRefusalJudge:
    def __init__(
        self,
        min_top1_score: float = 0.58,
        min_avg_top3_score: float = 0.52,
        min_score_gap: float = 0.03,
        ambiguous_top1_upper_bound: float = 0.65,
    ):
        self.min_top1_score = min_top1_score
        self.min_avg_top3_score = min_avg_top3_score
        self.min_score_gap = min_score_gap
        self.ambiguous_top1_upper_bound = ambiguous_top1_upper_bound

    def judge(self, chunks: List[RetrievedChunk]) -> RefusalDecision:
        if not chunks:
            return RefusalDecision(
                should_refuse=True,
                reason="no_chunks_retrieved",
                top1_score=0.0,
                top2_score=0.0,
                avg_top3_score=0.0,
            )

        top1 = chunks[0].score
        top2 = chunks[1].score if len(chunks) > 1 else 0.0
        top3_scores = [c.score for c in chunks[:3]]
        avg_top3 = sum(top3_scores) / len(top3_scores)

        if top1 < self.min_top1_score:
            return RefusalDecision(
                should_refuse=True,
                reason="top1_score_too_low",
                top1_score=top1,
                top2_score=top2,
                avg_top3_score=avg_top3,
            )

        if avg_top3 < self.min_avg_top3_score:
            return RefusalDecision(
                should_refuse=True,
                reason="avg_top3_score_too_low",
                top1_score=top1,
                top2_score=top2,
                avg_top3_score=avg_top3,
            )

        if len(chunks) > 1 and top1 < self.ambiguous_top1_upper_bound and (top1 - top2) < self.min_score_gap:
            return RefusalDecision(
                should_refuse=True,
                reason="top_results_ambiguous",
                top1_score=top1,
                top2_score=top2,
                avg_top3_score=avg_top3,
            )

        return RefusalDecision(
            should_refuse=False,
            reason=None,
            top1_score=top1,
            top2_score=top2,
            avg_top3_score=avg_top3,
        )