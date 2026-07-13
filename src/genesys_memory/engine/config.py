"""Centralized engine configuration.

All tunable thresholds in one place. Every value can be overridden via
environment variable. Defaults match the values validated against the
LoCoMo benchmark (89.9%) and the neuroscience literature (ACT-R, Rescorla-Wagner).
"""
from __future__ import annotations

import os


def _float(key: str, default: str) -> float:
    return float(os.getenv(key, default))


def _int(key: str, default: str) -> int:
    return int(os.getenv(key, default))


def _float_override(key: str) -> float | None:
    """Return the explicit env override for `key`, or None if unset."""
    val = os.getenv(key)
    return float(val) if val is not None else None


# ---------------------------------------------------------------------------
# ACT-R Scoring (scoring.py)
# ---------------------------------------------------------------------------
ACTR_DECAY_EXPONENT = _float("GENESYS_ACTR_DECAY", "0.5")
RELEVANCE_VECTOR_WEIGHT = _float("GENESYS_RELEVANCE_VECTOR_WEIGHT", "0.7")
RELEVANCE_KEYWORD_WEIGHT = _float("GENESYS_RELEVANCE_KEYWORD_WEIGHT", "0.3")
MIN_CONNECTIVITY = _float("GENESYS_MIN_CONNECTIVITY", "0.1")

# ---------------------------------------------------------------------------
# Status Transitions (transitions.py)
# ---------------------------------------------------------------------------
TAGGED_EXPIRE_HOURS = _int("GENESYS_TAGGED_EXPIRE_HOURS", "24")
ACTIVE_TO_EPISODIC_THRESHOLD = _float("GENESYS_ACTIVE_EPISODIC_THRESHOLD", "0.6")
ACTIVE_TO_EPISODIC_SESSIONS = _int("GENESYS_ACTIVE_EPISODIC_SESSIONS", "3")
DORMANCY_THRESHOLD = _float("GENESYS_DORMANCY_THRESHOLD", "0.15")
DORMANCY_DAYS = _int("GENESYS_DORMANCY_DAYS", "90")
DORMANCY_MAX_REACTIVATIONS = _int("GENESYS_DORMANCY_MAX_REACTIVATIONS", "3")

# ---------------------------------------------------------------------------
# Active Forgetting (forgetting.py)
# ---------------------------------------------------------------------------
FORGETTING_THRESHOLD = _float("GENESYS_FORGETTING_THRESHOLD", "0.01")

# ---------------------------------------------------------------------------
# Core Memory Promotion (promoter.py)
# ---------------------------------------------------------------------------
CORE_THRESHOLD = _float("GENESYS_CORE_THRESHOLD", "0.55")
CORE_ACTIVATION_WEIGHT = _float("GENESYS_CORE_ACTIVATION_WEIGHT", "0.4")
CORE_HUB_WEIGHT = _float("GENESYS_CORE_HUB_WEIGHT", "0.3")
CORE_SCHEMA_WEIGHT = _float("GENESYS_CORE_SCHEMA_WEIGHT", "0.2")
CORE_STABILITY_WEIGHT = _float("GENESYS_CORE_STABILITY_WEIGHT", "0.1")
AUTO_PROMOTE_CATEGORIES = [
    c.strip()
    for c in os.getenv(
        "GENESYS_AUTO_PROMOTE_CATEGORIES",
        "professional,educational,family,location",
    ).split(",")
]
HUB_SCORE_CAP = _float("GENESYS_HUB_SCORE_CAP", "3.0")
SCHEMA_NEIGHBOR_CAP = _int("GENESYS_SCHEMA_NEIGHBOR_CAP", "10")
STABILITY_CAP = _float("GENESYS_STABILITY_CAP", "3.0")

# ---------------------------------------------------------------------------
# Recall Relevance Filtering (tools.py — memory_recall)
# ---------------------------------------------------------------------------
# Cosine similarity floors are embedder-dependent: OpenAI's
# text-embedding-3-small produces well-separated similarities for genuine
# matches (~0.5+), but local sentence-transformers models and any
# other/unknown embedder (including test doubles with no embedder at all)
# cluster much lower, so a fixed OpenAI-tuned floor silently filters out
# true positives. These constants are the fallback defaults consulted by
# resolve_recall_min_similarity()/resolve_core_inject_min_similarity() below;
# an explicit GENESYS_RECALL_MIN_SIMILARITY / GENESYS_CORE_INJECT_MIN_SIMILARITY
# env var always wins over embedder-based defaults.
RECALL_MIN_SIMILARITY_OPENAI_DEFAULT = 0.5
RECALL_MIN_SIMILARITY_OTHER_DEFAULT = 0.2
CORE_INJECT_MIN_SIMILARITY_OPENAI_DEFAULT = 0.45
CORE_INJECT_MIN_SIMILARITY_OTHER_DEFAULT = 0.2

RECALL_MIN_SIMILARITY_OVERRIDE = _float_override("GENESYS_RECALL_MIN_SIMILARITY")
CORE_INJECT_MIN_SIMILARITY_OVERRIDE = _float_override("GENESYS_CORE_INJECT_MIN_SIMILARITY")

# Retained for backward compatibility with any external readers of the old
# single-value constants; reflects the OpenAI-tuned default (or the explicit
# env override). Prefer resolve_recall_min_similarity()/
# resolve_core_inject_min_similarity() which are embedder-aware.
RECALL_MIN_SIMILARITY = (
    RECALL_MIN_SIMILARITY_OVERRIDE
    if RECALL_MIN_SIMILARITY_OVERRIDE is not None
    else RECALL_MIN_SIMILARITY_OPENAI_DEFAULT
)
CORE_INJECT_MIN_SIMILARITY = (
    CORE_INJECT_MIN_SIMILARITY_OVERRIDE
    if CORE_INJECT_MIN_SIMILARITY_OVERRIDE is not None
    else CORE_INJECT_MIN_SIMILARITY_OPENAI_DEFAULT
)


def _embedder_recommended(embeddings: object | None, attr: str) -> float | None:
    """Read a `recommended_*_similarity` float off an embedding provider, if present.

    Uses getattr rather than a Protocol/isinstance check so any embedder
    (including test doubles) can opt in; non-numeric or missing values
    (e.g. Mock() attributes auto-created by AsyncMock-based test doubles)
    are treated as "no recommendation".
    """
    value = getattr(embeddings, attr, None)
    return float(value) if isinstance(value, (int, float)) else None


def resolve_recall_min_similarity(embeddings: object | None) -> float:
    """Effective memory_recall similarity floor.

    Precedence: explicit GENESYS_RECALL_MIN_SIMILARITY env var > the
    embedder's own `recommended_min_similarity` > the generic non-OpenAI
    default (0.2).
    """
    if RECALL_MIN_SIMILARITY_OVERRIDE is not None:
        return RECALL_MIN_SIMILARITY_OVERRIDE
    recommended = _embedder_recommended(embeddings, "recommended_min_similarity")
    return recommended if recommended is not None else RECALL_MIN_SIMILARITY_OTHER_DEFAULT


def resolve_core_inject_min_similarity(embeddings: object | None) -> float:
    """Effective core-memory-injection similarity floor.

    Precedence: explicit GENESYS_CORE_INJECT_MIN_SIMILARITY env var > the
    embedder's own `recommended_core_min_similarity` > the generic
    non-OpenAI default (0.2).
    """
    if CORE_INJECT_MIN_SIMILARITY_OVERRIDE is not None:
        return CORE_INJECT_MIN_SIMILARITY_OVERRIDE
    recommended = _embedder_recommended(embeddings, "recommended_core_min_similarity")
    return recommended if recommended is not None else CORE_INJECT_MIN_SIMILARITY_OTHER_DEFAULT

# ---------------------------------------------------------------------------
# Edge Staleness
# ---------------------------------------------------------------------------
EDGE_STALE_DAYS = _int("GENESYS_EDGE_STALE_DAYS", "30")
EDGE_STALE_PENALTY = _float("GENESYS_EDGE_STALE_PENALTY", "0.5")

# ---------------------------------------------------------------------------
# Cascade Reactivation (reactivation.py)
# ---------------------------------------------------------------------------
CASCADE_DEPTH = _int("GENESYS_CASCADE_DEPTH", "2")
CASCADE_DECAY_FACTOR = _float("GENESYS_CASCADE_DECAY_FACTOR", "0.3")
DORMANT_REVIVAL_THRESHOLD = _float("GENESYS_DORMANT_REVIVAL_THRESHOLD", "0.1")

# ---------------------------------------------------------------------------
# Ingestion Limits
# ---------------------------------------------------------------------------
MAX_INGEST_FILE_MB = _int("GENESYS_MAX_INGEST_FILE_MB", "100")
