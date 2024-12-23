# Import Libraries
from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np

from ragas.llms.prompt import Prompt
from ragas.metrics.base import MetricWithLLM
import json
from langchain_core.messages import HumanMessage

LONG_FORM_ANSWER_PROMPT = Prompt(
    name="long_form_answer",
    instruction="Create one or more statements from each sentence in the given answer.",
    examples=[
        {
            "question": "Who was  Albert Einstein and what is he best known for?",
            "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            "statements": {
                "statements": [
                    "Albert Einstein, a German-born theoretical physicist, is renowned for being one of the most influential physicists in history.",
                    "Albert Einstein was best known for his theory of relativity.",
                    "Einstein's contributions significantly advanced the field of quantum mechanics",
                    "Recognized globally, Einstein's work has profoundly impacted the scientific community",
                    "Einstein's groundbreaking theories continue to shape our understanding of physics today.",
                ]
            },
        },
        {
            "question": "Cadmium Chloride is slightly soluble in this chemical, it is also called what?",
            "answer": "alcohol",
            "statements": {
                "statements": ["Cadmium Chloride is slightly soluble in alcohol."]
            },
        },
        {
            "question": "Were Hitler and Benito Mussolini of the same nationality?",
            "answer": "Sorry, I can't provide answer to that question.",
            "statements": {"statements": []},
        },
    ],
    input_keys=["question", "answer"],
    output_key="statements",
    output_type="json",
)  # noqa: E501

NLI_STATEMENTS_MESSAGE = Prompt(
    name="nli_statements",
    instruction="Natural language inference. Use only 'Yes' (1), 'No' (0)",
    examples=[
        {
            "context": """John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
            "statements": """
            statement_1: John is majoring in Biology.
            statement_2: John is taking a course on Artificial Intelligence.
            statement_3: John is a dedicated student.
            statement_4: John has a part-time job.
            """,
            "answer": [
                {
                    "statement_1": "John is majoring in Biology.",
                    "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                    "verdict": "0",
                },
                {
                    "statement_2": "John is taking a course on Artificial Intelligence.",
                    "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                    "verdict": "0",
                },
                {
                    "statement_3": "John is a dedicated student.",
                    "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                    "verdict": "1",
                },
                {
                    "statement_4": "John has a part-time job.",
                    "reason": "There is no information given in the context about John having a part-time job.",
                    "verdict": "0",
                },
            ],
        },
        {
            "context": """Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.""",
            "statements": """statement_1: Albert Einstein was a genius.""",
            "answer": {
                "statement_1": "Albert Einstein was a genius.",
                "reason": "The context and statement are unrelated",
                "verdict": "0",
            },
        },
    ],
    input_keys=["context", "statements"],
    output_key="answer",
    output_type="json",
)  # noqa: E501

def _create_answer_prompt(row):
    question, answer = row["question"], row["answer"]

    # edit format
    question = question.replace('"', "'")
    answer = answer.replace('"', "'")
    # extract statements from answer given the question
    prompt_value = LONG_FORM_ANSWER_PROMPT.format(
        question=question, answer=answer
    )
    return prompt_value

def _create_nli_prompt(row, statements):
    contexts = row["contexts"]
    # check if the statements are support in the contexts
    contexts_str: str = "\n".join(contexts)
    statements_str: str = "\n".join(
        [f"statement_{i+1}: {st}" for i, st in enumerate(statements)]
    )
    prompt_value = NLI_STATEMENTS_MESSAGE.format(
        context=contexts_str, statements=statements_str
    )
    return prompt_value

def _compute_score(output):
    # check the verdicts and compute the score
    verdict_score_map = {"1": 1, "0": 0}
    output = output if isinstance(output, list) else [output]
    faithful_statements = sum(
        verdict_score_map.get(
            str(statement_with_validation.get("verdict", "")), np.nan
        )
        if isinstance(statement_with_validation, dict)
        else np.nan
        for statement_with_validation in output
    )
    num_statements = len(output)
    if num_statements:
        score = faithful_statements / num_statements
    else:
        print(
            "Invalid JSON response. Expected dictionary with key 'verdict'"
        )
        score = np.nan

    return score

def faithfulness(row, llm):
    if row['answer'] == '':
        return 0, [{}]
    
    # row['contexts'] = [item.replace('\n \n', '') for item in row['contexts']]

    p = _create_answer_prompt(row)
    # print("p: ", p)
    message = HumanMessage(content=p.prompt_str)
    # print("message: ", message)
    answer_result = llm([message])
    # print("answer_result.content: ", answer_result)
    # print(len(answer_result.content))
    # print(type(answer_result.content))
    print("cleaned: ", answer_result.content.strip('```json\n').strip('```'))
    # Edit to solve formatting issue
    # if answer_result.content[-1:] != '}':
    #     tmp = answer_result.content + '}'
    #     print("tmp: ", tmp)
    #     print("type: ", type(tmp))
    #     statements = json.loads(tmp)
    # else:
    #     statements = json.loads(answer_result.content)
    if len(answer_result.content) == 0:
        statements = {}
    else: 
        statements = json.loads(answer_result.content.strip('```json\n').strip('```')) # EDITED

    # print("statements: ", statements)
    statements = statements.get("statements", [])
    # print("len statements: ", len(statements))
    if statements:
        # LLM generate multiple times if the statements are too long: token limits
        if len(statements) > 5: # More than 5
            json_output = []
            for i in range(int(len(statements)/5)):
                p = _create_nli_prompt(row, statements[(i*5):(5*(i+1))])
                message = HumanMessage(content=p.prompt_str)
                nli_result = llm([message])
                print("nli_result.content 1: ", nli_result.content)

                ##new
                nli_result_content = nli_result.content.strip('```json\n').strip('```')

                # Data formatting issue
                if nli_result_content[0] != '[':
                    tmp = '[' + nli_result_content + ']'
                    tmp = tmp.replace('}\n{', '},\n{')
                    response = json.loads(tmp) 
                    json_output = json_output + response
                else:
                    json_output = json_output + json.loads(nli_result_content)
                ##

                # json_output = json_output + json.loads(nli_result.content.strip('```json\n').strip('```'))

            if 5*(i+1) < len(statements): # Last sequence
                p = _create_nli_prompt(row, statements[(5*(i+1)):])
                message = HumanMessage(content=p.prompt_str)
                nli_result = llm([message])
                print("nli_result.content 2: ", nli_result.content)

                nli_result_content = nli_result.content.strip('```json\n').strip('```')

                ### TODO MODIFY, CURRENT MAKESHIFT SOLUTION
                # if nli_result.content == '':
                #     json_output = json_output


                # Data formatting issue
                if nli_result_content[0] != '[':
                    tmp = '[' + nli_result_content + ']'
                    tmp = tmp.replace('}\n{', '},\n{')
                    response = json.loads(tmp) 
                    json_output = json_output + response
                else:
                    json_output = json_output + json.loads(nli_result_content)
                
                # json_output = json_output + json.loads(nli_result.content.strip('```json\n').strip('```'))

            # Process the statement id: increasing order
            for i in range(5, len(json_output)):
                old_key = list(json_output[i].keys())[0]
                new_key = 'statement_' + str(i+1)
                json_output[i][new_key] = json_output[i][old_key]
                del json_output[i][old_key]

        else:
            p = _create_nli_prompt(row, statements[:4])
            message = HumanMessage(content=p.prompt_str)
            nli_result = llm([message])

            nli_result_content = nli_result.content.strip('```json\n').strip('```')

            ## TODO MODIFY, CURRENT MAKESHIFT SOLUTION
            if nli_result.content == '':
                json_output = [{}]

            # Data formatting issue
            elif nli_result_content[0] != '[':
                tmp = '[' + nli_result_content + ']'
                tmp = tmp.replace('}\n{', '},\n{')
                print("tmp: ", tmp)
                response = json.loads(tmp) 
                json_output = response
            else:
                json_output = json.loads(nli_result_content)
            ##

    else:
        json_output = [{}]

    return _compute_score(json_output), json_output