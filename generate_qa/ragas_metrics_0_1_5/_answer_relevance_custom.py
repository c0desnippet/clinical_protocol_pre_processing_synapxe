from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np

from ragas.llms.prompt import Prompt
from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM
import json
from langchain_core.messages import HumanMessage
import re

QUESTION_GEN = Prompt(
    name="question_generation",
    # Edit prompts
    instruction="""Generate 3 different questions for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers.""",
    examples=[
        {
            "answer": """Albert Einstein was born in Germany.""",
            "context": """Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time""",
            "output": {
                "question": "Where was Albert Einstein born?",
                "noncommittal": 0,
            },
        },
        {
            "answer": """It can change its skin color based on the temperature of its environment.""",
            "context": """A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment.""",
            "output": {
                "question": "What unique ability does the newly discovered species of frog have?",
                "noncommittal": 0,
            },
        },
        {
            "answer": """Everest""",
            "context": """The tallest mountain on Earth, measured from sea level, is a renowned peak located in the Himalayas.""",
            "output": {
                "question": "What is the tallest mountain on Earth?",
                "noncommittal": 0,
            },
        },
        {
            "answer": """I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. """,
            "context": """In 2023, a groundbreaking invention was announced: a smartphone with a battery life of one month, revolutionizing the way people use mobile technology.""",
            "output": {
                "question": "What was the groundbreaking feature of the smartphone invented in 2023?",
                "noncommittal": 1,
            },
        },
    ],
    input_keys=["answer", "context"],
    output_key="output",
    output_type="json",
)


def _create_question_gen_prompt(row):
    ans, ctx = row["answer"], row["contexts"]
    return QUESTION_GEN.format(answer=ans, context="\n".join(ctx))


def calculate_similarity(question, generated_questions, embeddings):
    question_vec = np.asarray(embeddings.embed_query(question)).reshape(1, -1)
    gen_question_vec = np.asarray(
        embeddings.embed_documents(generated_questions)
    ).reshape(len(generated_questions), -1)
    norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
        question_vec, axis=1
    )
    return (
        np.dot(gen_question_vec, question_vec.T).reshape(
            -1,
        )
        / norm
    )


def _calculate_score(response, row, embeddings):
    question = row["question"]
    gen_questions = [
        item.get("question", "") for item in response if isinstance(item, dict)
    ]
    # gen_questions = [
    # q.get("question", "") for item in response if isinstance(item, dict)
    # for q in item.get("questions", []) if isinstance(q, dict)
    # ]
    print("gen_questions: ", gen_questions)
    committal = np.any(
        [
            bool(item.get("noncommittal", 0))
            for item in response
            if isinstance(item, dict)
        ]
    )
    if all(q == "" for q in gen_questions):
        score = np.nan
    else:
        # add embeddings
        cosine_sim = calculate_similarity(question, gen_questions, embeddings)
        score = cosine_sim.mean() * int(not committal)

    return score

# transform response to account for difference in output format
def transform_response(response):
    new_response = []
    for item in response:
        if isinstance(item, dict):
            questions = item.get('questions', [])
            if questions and isinstance(questions[0], dict):
                # Format 1: List of dictionaries
                new_response.extend(questions)
            elif questions and isinstance(questions[0], str):
                # Format 2: List of strings
                new_response.extend([{'question': q, 'noncommittal': item.get('noncommittal', 0)} for q in questions])
    return new_response

def answer_relevancy(row, llm, embeddings):
    prompt = _create_question_gen_prompt(row)
    message = HumanMessage(content=prompt.prompt_str)
    result = llm([message])
    # print("result.content: ", result.content)
    result_content = result.content.strip('```json\n').strip('```')
    
    # result_content = result.content.strip('```')
    result_content = result_content.replace('```json', '')
    result_content = result_content.replace('```', ',')

    # print("result_content: ", result_content)

    # print("result_content: ", len(result_content))
    
    # Edit the script to handle exception:
    if len(result_content) == 0:
        response = [{}]
    elif result_content[0] != '[':
        tmp = '[' + result_content + ']'
        # tmp = tmp.replace('}\n{', '},\n{')
        # print("tmp: ", tmp)
        tmp = re.sub('}[\n]+{', '},\n{', tmp)
        print('tmp: ', tmp)
        response = json.loads(tmp) 
    else:
        response = json.loads(result_content) 
    # # flatten response
    # response = [item for sublist in response for item in sublist]
    # transform response
    response = transform_response(response)
    score = _calculate_score(response, row, embeddings)

    return score, response



