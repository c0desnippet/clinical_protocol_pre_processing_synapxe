from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np

from ragas.llms.json_load import json_loader
from ragas.llms.prompt import Prompt
from ragas_metrics_0_1_5 import _answer_similarity_custom
from ragas.metrics.base import EvaluationMode, MetricWithEmbeddings, MetricWithLLM
import json
from langchain_core.messages import HumanMessage

# Prompt
CORRECTNESS_PROMPT = Prompt(
    name="answer_correctness",
    instruction="""Extract following from given question and ground truth
            "TP": statements that are present in both the answer and the ground truth,
            "FP": statements present in the answer but not found in the ground truth,
            "FN": relevant statements found in the ground truth but omitted in the answer, 
        """,
    examples=[
        {
            "question": """What powers the sun and what is its primary function?""",
            "answer": """The sun is powered by nuclear fission, similar to nuclear reactors on Earth, and its primary function is to provide light to the solar system.""",
            "ground_truth": """The sun is actually powered by nuclear fusion, not fission. In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy. This energy is what lights up the sun and provides heat and light, essential for life on Earth. The sun's light also plays a critical role in Earth's climate system and helps to drive the weather and ocean currents.""",
            "Extracted statements": {
                "TP": ["The sun's primary function is to provide light"],
                "FP": [
                    "The sun is powered by nuclear fission",
                    "similar to nuclear reactors on Earth",
                ],
                "FN": [
                    "The sun is powered by nuclear fusion, not fission",
                    "In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy",
                    "This energy provides heat and light, essential for life on Earth",
                    "The sun's light plays a critical role in Earth's climate system",
                    "The sun helps to drive the weather and ocean currents",
                ],
            },
        },
        {
            "question": """What is the boiling point of water?""",
            "answer": """The boiling point of water is 100 degrees Celsius at sea level.""",
            "ground_truth": """The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level, but it can change with altitude.""",
            "Extracted statements": {
                "TP": [
                    "The boiling point of water is 100 degrees Celsius at sea level"
                ],
                "FP": [],
                "FN": [
                    "The boiling point can change with altitude",
                    "The boiling point of water is 212 degrees Fahrenheit at sea level",
                ],
            },
        },
    ],
    input_keys=["question", "answer", "ground_truth"],
    output_key="Extracted statements",
    output_type="json",
)

def _compute_statement_presence(prediction):
    key_map = [
        "TP",
        "FP",
        "FN",
    ]
    prediction = prediction if isinstance(prediction, dict) else {}
    if prediction:
        prediction = [prediction.get(k, np.nan) for k in key_map]
        tp, fp, fn = [
            len(item) if isinstance(item, list) else np.nan for item in prediction
        ]
        if any([np.isnan(i) for i in [tp, fp, fn]]):
            score = np.nan
            logger.warning(
                "Invalid prediction format. Expected a list of dictionaries with keys 'TP', 'FP', 'FN'"
            )
        else:
            score = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0
    else:
        score = np.nan

    return score

def _create_correctness_prompt(row):
    q, a, g = row["question"], row["answer"], row["ground_truth"]
    p_value = CORRECTNESS_PROMPT.format(question=q, ground_truth=g, answer=a)
    return p_value

# Answer Correctness to output explanation
def answer_correctness(row, llm, embeddings, weights=[0.75, 0.25]):
    p_value = _create_correctness_prompt(row)

    # Score
    message = HumanMessage(content=p_value.prompt_str)
    is_statement_present = llm([message])
    
    # F1 score
    prediction = json.loads(is_statement_present.content)
    f1_score = _compute_statement_presence(prediction)

    # Weighted
    weights = weights

    # Similarity score
    if weights[1] == 0:
        similarity_score = 0
    else:
        similarity_score = _answer_similarity_custom.answer_similarity(row=row, embeddings=embeddings)

    
    score = np.average(
            [f1_score, similarity_score],
            weights=weights,
        )
    
    return score, prediction