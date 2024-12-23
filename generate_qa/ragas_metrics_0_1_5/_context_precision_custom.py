from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset

from ragas.llms.json_load import json_loader
from ragas.llms.prompt import Prompt, PromptValue
from ragas.metrics.base import EvaluationMode, MetricWithLLM
import json
from langchain_core.messages import HumanMessage

CONTEXT_PRECISION = Prompt(
    name="context_precision",
    instruction="""Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output. """,
    examples=[
        {
            "question": """What can you tell me about albert Albert Einstein?""",
            "context": """Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.""",
            "answer": """Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895""",
            "verification": {
                "reason": "The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.",
                "verdict": "1",
            },
        },
        {
            "question": """who won 2020 icc world cup?""",
            "context": """The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.""",
            "answer": """England""",
            "verification": {
                "reason": "the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",
                "verdict": "1",
            },
        },
        {
            "question": """What is the tallest mountain in the world?""",
            "context": """The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.""",
            "answer": """Mount Everest.""",
            "verification": {
                "reason": "the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.",
                "verdict": "0",
            },
        },
    ],
    input_keys=["question", "context", "answer"],
    output_key="verification",
    output_type="json",
)

def _get_row_attributes(row):
    answer = "ground_truth"
    if answer not in row.keys():
        answer = "answer"
    return row["question"], row["contexts"], row[answer]

def _context_precision_prompt(row):
    question, contexts, answer = _get_row_attributes(row)
    # edit format
    question = question.replace('"', "'")
    answer = answer.replace('"', "'")
    contexts = [txt.replace('"', "'") for txt in contexts]
    return [CONTEXT_PRECISION.format(
            question=question, context=c, answer=answer
        )
        for c in contexts
    ]

def _calculate_average_precision(json_responses):
    score = np.nan
    json_responses = [
        item if isinstance(item, dict) else {} for item in json_responses
    ]
    verdict_list = [
        int("1" == resp.get("verdict", "").strip())
        if resp.get("verdict")
        else np.nan
        for resp in json_responses
    ]
    denominator = sum(verdict_list) + 1e-10
    numerator = sum(
        [
            (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
            for i in range(len(verdict_list))
        ]
    )
    score = numerator / denominator
    return score


def context_precision(row, llm):
    human_prompts = _context_precision_prompt(row)
    responses = []
    for hp in human_prompts:
        message = HumanMessage(content=hp.prompt_str)
        result = llm([message])
        responses.append(result.content)

    json_responses = [json.loads(i) for i in responses]
    json_responses = t.cast(t.List[t.Dict], json_responses)
    score = _calculate_average_precision(json_responses)

    return score, json_responses