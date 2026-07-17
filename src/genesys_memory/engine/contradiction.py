"""Contradiction detection between memories."""
from __future__ import annotations

import re

from genesys_memory.models.edge import MemoryEdge
from genesys_memory.models.enums import EdgeType, MemoryStatus
from genesys_memory.models.node import MemoryNode
from genesys_memory.storage.base import EmbeddingProvider, GraphStorageProvider, LLMProvider

_NEGATION_RE = re.compile(r"\b(not|never|no longer|n't|isn't|won't|doesn't)\b", re.IGNORECASE)
_NUM_OR_WORD_RE = re.compile(r"[a-zA-Z]+|\d[\d,.]*%?")


def _number_contexts(text: str) -> dict[str, set[str]]:
    """Map each number token to its context: the nearest preceding word.

    ``"budget is 50k"`` → ``{"is": {"50"}}`` (with the number token as found).
    Numbers with no preceding word key on ``""``.
    """
    contexts: dict[str, set[str]] = {}
    last_word = ""
    for tok in _NUM_OR_WORD_RE.findall(text):
        if tok[0].isdigit():
            contexts.setdefault(last_word.lower(), set()).add(tok)
        else:
            last_word = tok
    return contexts


def heuristic_conflict_signal(text_a: str, text_b: str) -> str | None:
    """Cheap, dependency-free lexical divergence check between two texts.

    Returns ``"numeric_mismatch"`` when the two texts mention *differing*
    numbers in a comparable position — i.e. some shared nearest-preceding
    context word carries different number sets in each text ("costs 50" vs
    "costs 75"). Numbers in unrelated positions (a date in one text, an ID in
    the other) do NOT fire, which keeps the hint from being noise on realistic
    workloads. Returns ``"negation"`` when exactly one text is negated;
    otherwise ``None``. These are hints only — never verified contradictions
    and never materialized as edges.
    """
    ctx_a = _number_contexts(text_a)
    ctx_b = _number_contexts(text_b)
    for key in ctx_a.keys() & ctx_b.keys():
        if ctx_a[key] != ctx_b[key]:
            return "numeric_mismatch"
    neg_a = bool(_NEGATION_RE.search(text_a))
    neg_b = bool(_NEGATION_RE.search(text_b))
    if neg_a != neg_b:
        return "negation"
    return None


async def detect_contradictions(
    new_node: MemoryNode,
    graph: GraphStorageProvider,
    embeddings: EmbeddingProvider,
    llm: LLMProvider,
) -> list[tuple[str, float]]:
    """
    Check if new_node contradicts existing memories.
    1. Vector search for similarity > 0.85
    2. LLM confirmation
    3. Create CONTRADICTS edges for confirmed contradictions
    Returns list of (contradicted_node_id, confidence).
    """
    if not new_node.embedding:
        return []

    # Find highly similar memories (potential contradictions)
    candidates = await graph.vector_search(new_node.embedding, k=20)
    contradictions: list[tuple[str, float]] = []

    for candidate_node, sim_score in candidates:
        # Skip self
        if str(candidate_node.id) == str(new_node.id):
            continue
        # Only check high-similarity pairs
        # FalkorDB cosine distance: lower = more similar; convert to similarity
        similarity = 1.0 - sim_score if sim_score <= 1.0 else sim_score
        if similarity < 0.85:
            continue

        content_a = new_node.content_full or new_node.content_summary
        content_b = candidate_node.content_full or candidate_node.content_summary
        is_contradiction, confidence, reason = await llm.detect_contradiction(content_a, content_b)

        if is_contradiction and confidence > 0.7:
            # Create CONTRADICTS edge
            edge = MemoryEdge(
                source_id=new_node.id,
                target_id=candidate_node.id,
                type=EdgeType.CONTRADICTS,
                weight=confidence,
                reason=reason,
                created_by="llm_contradiction",
            )
            await graph.create_edge(edge)
            contradictions.append((str(candidate_node.id), confidence))

            # If contradicted memory is core, trigger supersession
            if candidate_node.status == MemoryStatus.CORE:
                supersede_edge = MemoryEdge(
                    source_id=new_node.id,
                    target_id=candidate_node.id,
                    type=EdgeType.SUPERSEDES,
                    weight=confidence,
                    reason=reason,
                    created_by="llm_contradiction",
                )
                await graph.create_edge(supersede_edge)
                await graph.update_node(str(candidate_node.id), {
                    "status": MemoryStatus.EPISODIC,
                })

    return contradictions
