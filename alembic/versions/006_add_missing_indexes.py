"""Add missing indexes for edge queries, embedding_384, and stability.

Revision ID: 006
Revises: 005
Create Date: 2026-05-13
"""
from typing import Sequence, Union
from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # HNSW index on 384-dim embeddings (matches the 1536-dim index from 001)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodes_embedding_384 ON memory_nodes
        USING hnsw (embedding_384 vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    """)

    # Edge queries by user + type (causal chain traversal, edge type filtering)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_user_type ON memory_edges(user_id, type)"
    )

    # Edge queries scoped to a single user
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_edges_user_id ON memory_edges(user_id)"
    )

    # Stability lookups and sorting (core promotion scoring)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_nodes_stability ON memory_nodes(stability)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_nodes_stability")
    op.execute("DROP INDEX IF EXISTS idx_edges_user_id")
    op.execute("DROP INDEX IF EXISTS idx_edges_user_type")
    op.execute("DROP INDEX IF EXISTS idx_nodes_embedding_384")
