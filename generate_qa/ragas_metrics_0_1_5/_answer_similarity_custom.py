from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import numpy as np

from ragas.embeddings.base import HuggingfaceEmbeddings
from ragas.metrics.base import EvaluationMode, MetricWithEmbeddings, MetricWithLLM
import json
from langchain_core.messages import HumanMessage

def answer_similarity(row, embeddings):
    ground_truth = t.cast(str, row["ground_truth"])
    answer = t.cast(str, row["answer"])

    embedding_1 = np.array(embeddings.embed_query(ground_truth))
    embedding_2 = np.array(embeddings.embed_query(answer))
    # Normalization factors of the above embeddings
    norms_1 = np.linalg.norm(embedding_1, keepdims=True)
    norms_2 = np.linalg.norm(embedding_2, keepdims=True)
    embedding_1_normalized = embedding_1 / norms_1
    embedding_2_normalized = embedding_2 / norms_2
    similarity = embedding_1_normalized @ embedding_2_normalized.T
    score = similarity.flatten()
    return score[0]