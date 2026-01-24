"""
Task Embeddings subpackage - Task embedding utilities.

Contains:
- learnable_task_embeddings: LearnableTaskEmbedding class
"""

from ct2echo.task_embeddings.learnable_task_embeddings import (
    LearnableTaskEmbedding,
    create_learnable_task_embeddings,
    create_learnable_task_embedding_dict,
)

__all__ = [
    "LearnableTaskEmbedding",
    "create_learnable_task_embeddings",
    "create_learnable_task_embedding_dict",
]
