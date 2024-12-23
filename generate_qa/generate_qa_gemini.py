"""Generate QA Data_GEMINI_v1.py"""

# Variables
LOGSFOLDER = 'QA_Gen_Logs'
DOC_TITLE = 'When to order MRI for low back pain'
DATAPATH = f'../generate_qa/'
INPUTFILE = 'When to order MRI for low back pain.xlsx'
OUTPUTFILE = 'When to order MRI for low back pain_v2.csv'
TEMPERATURE = 0.0

specialchar_replacements = {'\u2265': ' more than or equals to ', '\u2264': ' less than or equals to ',
                            '>': ' more than ', '<': ' less than '}


"""# Imports"""
from ragas_metrics_0_1_5 import _faithfulness_custom
from ragas_metrics_0_1_5 import _answer_relevance_custom
from ragas_metrics_0_1_5 import _context_precision_custom
from ragas_metrics_0_1_5 import _context_recall_custom
from ragas_metrics_0_1_5 import _answer_similarity_custom
from ragas_metrics_0_1_5 import _answer_correctness_custom
import os
import pandas as pd
import numpy as np
import time
import pickle
import re
import logging
from datetime import datetime
import google.generativeai as genai
from datasets import Dataset


# Authenticate and build the gemini API client
'''
Requires user specific inputs.

Replace GEMINI_API_KEY with your gemini api key obtained.
'''
GEMINI_API_KEY = ''
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Setup logging
logging.basicConfig(filename=f'{LOGSFOLDER}/app_{datetime.now().strftime("%Y%m%d")}.log', 
                    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Functions
def generate_qa(doc_chunk, title=DOC_TITLE):
    """
    Generates at least twenty unique question-answer pairs based on the given document chunk using a template. 
    The questions and answers are derived from the source content and should adhere to a set of rules regarding clarity, brevity, and relevance.

    Args:
        doc_chunk (dict): A dictionary containing 'text_chunk', 'section_name', and 'pages' from the document.
        title (str): The title of the document (default is 'When to order MRI for low back pain').

    Returns:
        tuple: A tuple containing the generated QA text, page numbers, and section name.
    """
    source = doc_chunk['text_chunk']
    section_name = doc_chunk['section_name']
    pages = doc_chunk['pages']

    
    """# Setup

    ### Prompts
    """

    template_protocol = f"""
                        Source:
                        {section_name}
                        {source}

                        You are a professional doctor who assist to generate questions and answers related to clinical protocols.
                        Your task is to formulate at least twenty unique question-answer pairs based on the Source given above, satisfying the rules below:
                        1. The generated questions should be different in meanings but relevant to healthcare, on topics such as
                        Diagnostic or cut-off criteria, Dosage of medication and treatment guidelines based on patient group, Best practices for specific conditions,
                        Specialist or HCM to refer to patient of certain conditions to, Recommended resources or articles related to the management of certain condition,
                        Protocol or order set for certain condition, Possible drug-drug interactions or contraindications to any prescription.
                        2. The generated questions should be independent from the Source but with answers that can be answered using information from the Source.
                        3. Prioritize clarity and brevity, ensuring that the questions are complete and comprehensive to clinicians.
                        4. Where applicable, include grade of recommendation and level of evidence in the response.
                        5. Only generate questions and answers that can be derived from the given Source.
                        6. Ensure uniqueness and non-repetition in all the questions.
                        7. The questions should be diverse and comprehensively cover all the content of the source.
                        8. Keep count of the number of questions generated. Ensure that there are at least twenty unique generated questions. If there are fewer than twenty questions, use the above rules to generate more questions. There should be at least twenty unique questions generated.
                        9. Output the question-answer pairs in the exact format as the examples provided below.
                        10. Here are some examples:

                        **Question 1:** What are possible drug drug interactions to look out for when using ACE inhibitor?
                        **Answer:** Drug interactions to look out for include:
                        a) Potassium (K+) supplements/K+-sparing diuretics, e.g. amiloride and triamterene (including combination preparations with furosemide)
                        b) Nonsteroidal anti-inflammatory drugs
                        c) Low salt substitutes with high K+ content

                        **Question 2:** What are common comorbidities seen in patients with HFpEF?
                        **Answer:** Noncardiovascular comorbidities frequently seen in patients with Heart failure with preserved ejection fraction (HFpEF) are renal impairment, chronic lung diseases, anaemia, cancer, and peptic ulcer disease

                        **Question 3:** Should arterial blood gas (ABG) be performed in patients with heart failure?
                        **Answer:** Arterial blood gas should not be measured routinely in patients with haemodynamically stable acute decompensated heart failure. (Grade C, Level 2+). Arterial blood gas may be measured in select patients with acute decompensated heart failure if is an increased respiratory rate of >22% or oxygen saturation <92%, despite high flow inspired oxygen (>8L/min) or ventilator support is under consideration. (Grade B, Level 1+). Venous blood gas may be measured in select patients with acute decompensated heart failure as an alternative to arterial blood gas if there is risk of vascular injury. (Grade B, Level 1+)

                        11. REMEMBER, Do NOT generate questions and answers that cannot be derived from the Source.

                        """

    start = time.time()
    try:
        response = model.generate_content(template_protocol)
        if hasattr(response, 'text') and response.text:
            end = time.time()
            print(f'\nInference Time: {round(end - start, 3)}s')
            print('-' * 20, 'Output', '-' * 20)
            print(f"\nResponse from Model:\n{response.text}")
            return response.text, pages, section_name
        else:
            return "", pages, section_name
    except Exception as e:
        print(f"Error generating QA: {e}")
        return "", pages, section_name

def extract_qa(response_text):
    """
    Extracts question-answer pairs from the response text generated by the Gemini model.
    The pairs are identified using regular expressions and formatted by removing extra characters.

    Args:
        response_text (str): The text containing generated questions and answers.

    Returns:
        list: A list of tuples containing question and answer pairs.
    """
    qa_list = []

    # Use regex to find all question and answer pairs
    qa_pairs = re.findall(r'(\*?\*?Question .*?\*?\*?.*?\n\*?\*?Answer\*?\*?.*?)(?=\n\*?\*?Question|\Z)', response_text, re.DOTALL)

    # Process each pair
    for pair in qa_pairs:
        question = re.search(r'(\*?\*?Question.*?\*?\*?.*?)(?=\n\*?\*?Answer)', pair, re.DOTALL).group(1).strip()
        answer = re.search(r'(\*?\*?Answer\*?\*?.*)', pair, re.DOTALL).group(1).strip()

        # Remove the '**Question X:**' and '**Answer:**' parts, including the question number
        question = re.sub(r'\*?\*?Question\s*\d*:\*?\*?', '', question).strip()
        answer = re.sub(r'\*?\*?Answer\*?\*?:', '', answer).strip()

        # Remove any remaining asterisks
        question = re.sub(r'\*', '', question).strip()
        answer = re.sub(r'\*', '', answer).strip()

        qa_list.append((question, answer))

    return qa_list


def get_context(question, answer, source):
    """
    Extracts relevant context from the source document for a given question-answer pair by generating content using the Gemini model.

    Args:
        question (str): The generated question.
        answer (str): The generated answer.
        source (str): The source document from which the question and answer are derived.

    Returns:
        str: A string containing the extracted context related to the question-answer pair.
    """
    # Construct the template input
    template_context = f"""
    Sources:
    ref_context: {source}
    Question: {question}
    Answer: {answer}

    You are a professional doctor who assists in extracting the relevant context for questions and answers related to clinical protocols.
    Your task is to carefully examine the question and answer provided in the source above, and extract the relevant part of the ref_context also given above,
    where the question and answer are inferred from. The extracted part should include neighbouring context, and the extracted part should be about two to three sentences long.
    Avoid returning empty responses.

    Provide the response by returning the extracted portion as the 'Context'.

    Use simple language in the questions generated that are accessible to a broad audience.

    Format the response as:
    Context: ...
    """
    try:
        # Call the Gemini model to generate content
        response = model.generate_content(template_context)

        # Check if the response contains valid text
        if hasattr(response, 'text') and response.text:
            # Strip unnecessary whitespace and return the result
            return response.text.strip()
        else:
            # Return empty string if no valid response is found
            return ""
    except Exception as e:
        # Print the error message and return an empty string
        print(f"Error generating context: {e}")
        return ""

# Read the .xlsx file into a DataFrame
"""
Reads an Excel file containing clinical protocol data, processes it, and generates a list of question-answer pairs based on the content.
The questions are extracted using the `generate_qa` function, and the results are saved in a temporary CSV file for evaluation.

Requires user specific inputs.

Edit the sheet_name accordingly to the one that contains the reviewed text chunks
"""
df = pd.read_excel(DATAPATH + INPUTFILE, sheet_name='Copy of Sheet1')

# Save the DataFrame as a pickle file
with open('Final_chunks.pkl', 'wb') as f:
    pickle.dump(df, f)

"""# Read Data"""
with open('Final_chunks.pkl', 'rb') as fp:
    data = pickle.load(fp)

df = pd.DataFrame(data)
df.head()

# remove those without section name (between title and first section header)
df = df[~df['section_name'].isna()]

# Replace special characters
df['text_chunk'] = df['text_chunk'].replace(specialchar_replacements, regex=True)
df[df['text_chunk'].str.len()<100]

# keep only those chunks with at least 100 characters
filtered_df = df[df['text_chunk'].str.len() > 100]
filtered_df

'''
Requires user specific inputs.

Edit the range for the filtered_df accordingly. Currently it only generates QA for 1 row of text chunk.
'''
filtered_df = filtered_df.iloc[8:10]

# Generate QA pairs
start = time.time()
qa_dataset = []
response_all = []

for _, row in filtered_df.iterrows():
    time.sleep(30)
    response_text, pages, section = generate_qa(row)
    print(response_text)

    response_all.append(response_text)

    qa_list = extract_qa(response_text)
    print(len(qa_list))

    doc_chunk = row['text_chunk']

    for qa_pair in qa_list:
        question = qa_pair[0]
        answer = qa_pair[1]
        reference = f"For more information, refer to clinical protocol {DOC_TITLE}, Section {section}, Pages {pages}"


        qa_dataset.append({'question': question,
                           'answer': answer,
                           'full_text': response_text,
                           'reference': reference,
                           'section': section,
                           'pages': pages,
                           'doc_chunk': doc_chunk,
                           })


    tmp = pd.DataFrame(qa_dataset)
    tmp.to_csv("QA_tmp.csv", index=False)

end = time.time()
print(f'\nTotal Time: {round(end - start, 3)}s')

model_output = pd.DataFrame(qa_dataset)
model_output['Title'] = DOC_TITLE

"""# Checks & Eval"""

keywords = ['the source', 'this information', 'these guidelines', 'this source']

# Manual check for keywords
model_output['Flag_Source'] = np.where(model_output['question'].str.lower().str.contains('|'.join(keywords)),
                                1, 0)

model_output.head()

model_output[model_output['Flag_Source']==1]

"""### Run Faithfulness"""
# format data
model_output = model_output.rename(columns={"doc_chunk":"contexts"})
model_output = model_output[~model_output['contexts'].isna()]
model_output['contexts'] = model_output['contexts'].replace({'"': '\''}, regex=True)
model_output = Dataset.from_pandas(model_output)
print(model_output.shape)
model_output[0]

eval_output = model_output.to_pandas()
eval_output.head()

# run faithfulness
start_point = 0
eval_df = pd.DataFrame()

for i in range(start_point, len(model_output)):
    if model_output['Flag_Source'] == 1:
        faithfulness_result = [np.nan, np.nan]
    
    else:
    # faithfulness
        print(f"model_output: {model_output[i]}")
        faithfulness_result = _faithfulness_custom.faithfulness(model_output[i], model=model)
        print(f"faithfulness_result: {faithfulness_result}")


    tmp = pd.DataFrame({
                        'faithfulness': faithfulness_result[0],
                        'faithfulness_reasons': str(faithfulness_result[1]),

                        }, index=[0])
    eval_df = pd.concat([eval_df, tmp], ignore_index=True)
    output = pd.concat([eval_output[start_point:i+1], eval_df], axis=1)
    output.to_csv("eval_tmp.csv", index=False)

output = pd.concat([eval_output[start_point:i+1].reset_index(drop=True), eval_df.reset_index(drop=True)], axis=1)
output.to_csv(DATAPATH+OUTPUTFILE, index=False)