import json
import pandas as pd
import pickle
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.http import MediaIoBaseUpload
from io import BytesIO


# Authenticate and build the google Drive API client
'''
Requires user specific inputs

Replace SERVICE_ACCOUNT_FILE with the path to your JSON key file.
'''
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = ''
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=creds)

'''
TO EDIT KEEPTEXT and REFERENCETEXT and MAX_TEXT_CHAR

1. KEEPTEXT are basically the "Text" fields*** that you want to be output as section_name and not section content 
***these text do not fall under the definition of section headers in split_using_pathheader function, hence KEEPTEXT is required to keep them as section names.

2. REFERENCETEXT is the "Text" that will stop the split_using_pathheader function and stop text chunking

3. MAX_TEXT_CHAR is the maximum number of characters in one chunk
'''
REFERENCETEXT = "References"
KEEPTEXT = []
MAX_TEXT_CHAR = 3000

# to replace special characters
specialchar_replacements = {'≥': 'more than or equals to', '≤': 'less than or equals to'}

# text splitter to use for large chunks within sections
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_TEXT_CHAR,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# Find for the folder and return folder id, if not found create and return folder id
def get_or_create_folder_in_drive(parent_folder_id, folder_name):
    """
    Checks if a folder with a specific name exists in the parent folder in Google Drive.
    If it exists, returns the folder ID. Otherwise, creates the folder and returns its ID.

    Args:
        parent_folder_id (str): The ID of the parent folder.
        folder_name (str): The name of the folder to check for or create.

    Returns:
        str: The ID of the existing or newly created folder.
    """
    # Define the query to check for the folder
    query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='{folder_name}'"
    
    try:
        # Check if the folder exists
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        items = results.get('files', [])
        
        if items:
            print(f"Folder '{folder_name}' already exists with ID: {items[0]['id']}")
            return items[0]['id']
        
        # If the folder doesn't exist, create it
        else:
            print(f"Folder '{folder_name}' does not exists, creating now")
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_folder_id]
            }
            folder = service.files().create(body=folder_metadata, fields='id').execute()
            print(f"Folder '{folder_name}' created with ID: {folder.get('id')}")
            return folder.get('id')
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# New function to list folders
def list_folders_in_folder(folder_id):
    """
    Lists all subfolders within a specified Google Drive folder.

    Args:
        folder_id (str): The ID of the Google Drive folder whose subfolders are to be listed.

    Returns:
        list: A list of dictionaries, each representing a subfolder with its metadata (e.g., 'name' and 'id').
    """
    query = f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'"
    results = service.files().list(q=query).execute()
    return results.get('files', [])

# New function to list files
def list_files_in_folder(folder_id):
    """
    Lists all files (excluding subfolders) within a specified Google Drive folder.

    Args:
        folder_id (str): The ID of the Google Drive folder whose files are to be listed.

    Returns:
        list: A list of dictionaries, each representing a file with its metadata (e.g., 'name' and 'id').
    """
    query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"
    results = service.files().list(q=query).execute()
    return results.get('files', [])

# Function to upload files
def upload_file(service, file_name, mime_type, file_data, parent_folder_id):
    """
    Uploads a file to Google Drive under a specified parent folder.

    Args:
        service (googleapiclient.discovery.Resource): The authenticated Google Drive API service.
        file_name (str): The name to assign to the uploaded file.
        mime_type (str): The MIME type of the file to upload (e.g., 'application/pdf', 'text/plain').
        file_data (bytes): The binary content of the file to upload.
        parent_folder_id (str): The ID of the parent folder in Google Drive.

    Returns:
        str: The ID of the uploaded file.
    """
    file_metadata = {
        'name': file_name,
        'mimeType': mime_type,
        'parents': [parent_folder_id]
    }
    
    media = MediaIoBaseUpload(io.BytesIO(file_data), mimetype=mime_type)
    
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'Uploaded {file_name} with ID: {file.get("id")}')
    return file.get('id')


# Function to download files
def download_file_from_drive(file_id, destination_path):
    """
    Downloads a file from Google Drive and saves it to the local system.

    Args:
        file_id (str): The ID of the Google Drive file to download.
        destination_path (str): The local path where the downloaded file will be saved.

    Returns:
        None
    """
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}% complete.")
    print(f"File downloaded to {destination_path}.")

# Function to extract element from string
def extract_element(text, to_match="Table"):
    """
    Extracts a specific element from a string based on a regular expression.

    Args:
        text (str): The text from which the element is to be extracted.
        to_match (str): The element to match within the text (default is "Table").

    Returns:
        str or None: The matched element from the text, or None if no match is found.
    """
    pattern = re.compile(rf"//Document/{to_match}(\[\d+\])?/")
    match = pattern.search(text)
    return match.group() if match else None

# All other functions related to text chunking (split_using_pathheader, process_extract, etc.) remain unchanged
def split_using_pathheader(inputs_elements_df, referencetext=REFERENCETEXT, keepexceptiontext=KEEPTEXT):
    """
    Splits and categorizes text elements from an input DataFrame based on their paths and attributes.

    Args:
        inputs_elements_df (pd.DataFrame): A DataFrame containing text elements with attributes such as 'Path', 'Text', 'Page', etc.
        referencetext (str): The reference text used to identify the stopping condition (default is REFERENCETEXT).
        keepexceptiontext (list): A list of texts that should be kept as exception sections (default is KEEPTEXT).

    Returns:
        tuple: A tuple containing categorized data:
            - title (list): Extracted title information.
            - tables (list): Extracted table metadata.
            - figures (list): Extracted figure metadata.
            - text_chunks (list): Extracted text chunks.
            - sections (list): Extracted section headers.
            - exception_section (list): Sections flagged as exceptions.
    """
    elements_df = inputs_elements_df.copy()

    # clean up extracted data
    title = []
    title_count = 0
    tables = []
    tables_count = 0
    figures = []
    figures_count = 0
    sections = []
    sections_count = 0
    text_chunks = []
    text_chunks_count = 0
    exception_section = []

    for ind, row in elements_df.iterrows():
        if isinstance(row['Text'], str):
            '''
            TO EDIT

            1. path used to identify title
            2. edit second elif condition to suit the characteristics of the path attribute for the majority section names
            '''
            # edit path used to identify title 
            if ("//Document/Figure" == row['Path']):
                title_count +=1
                tmp_dict = {'title_id': title_count, 'title_name': row['Text'], 'Path': row['Path'], 'Page': row['Page']+1, 'ObjectID': row['ObjectID']} # page need to add 1 because start counting from 0}
                title.append(tmp_dict)

            elif row['Text'].strip() in keepexceptiontext:
                    sections_count += 1
                    tmp_dict = {'section_id': sections_count, 'section_name': row['Text'], 'Path': row['Path'], 'Page': row['Page']+1, 'ObjectID': row['ObjectID']} # page need to add 1 because start counting from 0}
                    sections.append(tmp_dict)
                    exception_section.append(tmp_dict)
                    print(f"Found {keepexceptiontext}, section_id {sections_count}, ObjectID {row['ObjectID']}.")

            # edit this elif condition to suit the characteristics of the path attribute for the majority section names
            elif "/H1" in row['Path'] and row['Text'].strip() != "www.ace-hta.gov.sg":
                # check if section is under References, stop function if so
                if row['Text'].strip() == referencetext:
                    print(f"Hit {referencetext}, ending function")
                    return title, tables, figures, text_chunks, sections, exception_section
            
                sections_count += 1
                tmp_dict = {
                    'section_id': sections_count,
                    'section_name': row['Text'],
                    'Path': row['Path'],
                    'Page': row['Page'] + 1,
                    'ObjectID': row['ObjectID']
                }
                sections.append(tmp_dict)

            else: # not section headers
                text_chunks_count += 1
                # check if text is part of table / figure
                element = extract_element(row['Path'], "Table")
                element = extract_element(row['Path'], "Figure")
                tmp_dict = {'text_id': text_chunks_count, 'section_id': sections_count, 'Path': row['Path'], 'Text': row['Text'], 'Page': row['Page']+1, 'ObjectID': row['ObjectID'], 'Add_Element': element} # page need to add 1 because start counting from 0
                text_chunks.append(tmp_dict)

        else: # those with text is na could be tables or figures, save them
            if "/Table" in row['Path']:
                pattern = re.compile(r'^//Document/Table(?:\[\d+\])?$')
                match = pattern.match(row['Path'])
                if match:
                    tables_count += 1
                    tmp_dict = {'table_id': tables_count, 'Path': row['Path'], 'Page': row['Page']+1, 'filePath': row['filePaths'], 'ObjectID': row['ObjectID']} # page need to add 1 because start counting from 0}
                    tables.append(tmp_dict)
            elif "/Figure" in row['Path']:
                pattern = re.compile(r'^//Document/Figure(?:\[\d+\])?$')
                match = pattern.match(row['Path'])
                if match:
                    figures_count += 1
                    tmp_dict = {'figure_id': figures_count, 'Path': row['Path'], 'Page': row['Page']+1, 'filePath': row['filePaths'], 'ObjectID': row['ObjectID']} # page need to add 1 because start counting from 0}
                    figures.append(tmp_dict)

    return title, tables, figures, text_chunks, sections, exception_section

def process_extract(text_chunks, sections):
    # Second step: combine all the texts within the same section
    chunks_data = []
    grouped_texts = defaultdict(str)
    grouped_pages = defaultdict(list)
    grouped_ids = defaultdict(list)

    # Group the text chunks by section
    for entry in text_chunks:
        section_id = entry['section_id']
        text = entry['Text']
        page = entry['Page']
        textid = entry['text_id']

        # Add a line space between concatenated texts
        if grouped_texts[section_id]:
            grouped_texts[section_id] += "\n\n"  # Add a line space

        grouped_texts[section_id] += text
        grouped_pages[section_id].append(page)
        grouped_ids[section_id].append(textid)

    # Create section lookup
    lookup_dict = {d['section_id']: d['section_name'] for d in sections}

    # Create chunks data for every section, even if there's no text
    for section in sections:
        section_id = section['section_id']
        section_name = lookup_dict.get(section_id, None)
        concatenated_text = grouped_texts.get(section_id, "")

        if len(concatenated_text) > MAX_TEXT_CHAR:
            # Split the concatenated text into semantic chunks
            split_chunks = text_splitter.split_text(concatenated_text)
            for chunk in split_chunks:
                tmp_dict = {
                    'text_chunk': chunk,
                    'section_name': section_name,
                    'text_id': grouped_ids.get(section_id, []),
                    'pages': list(set(grouped_pages.get(section_id, [])))
                }
                chunks_data.append(tmp_dict)
        else:
            tmp_dict = {
                'text_chunk': concatenated_text,
                'section_name': section_name,
                'text_id': grouped_ids.get(section_id, []),
                'pages': list(set(grouped_pages.get(section_id, [])))
            }
            chunks_data.append(tmp_dict)

    return chunks_data


# Process and combine text chunks based on sections
def process_extract(text_chunks, sections):
    """
    Processes text chunks and combines them into sections based on their IDs, ensuring semantic chunking when necessary.

    Args:
        text_chunks (list): A list of dictionaries representing individual text chunks.
        sections (list): A list of dictionaries representing sections with IDs and names.

    Returns:
        list: A list of dictionaries representing processed and combined text chunks for each section.
    """
    # Second step: combine all the texts within the same section
    chunks_data = []
    grouped_texts = defaultdict(str)
    grouped_pages = defaultdict(list)
    grouped_ids = defaultdict(list)

    # Group the text chunks by section
    for entry in text_chunks:
        section_id = entry['section_id']
        text = entry['Text']
        page = entry['Page']
        textid = entry['text_id']

        # Add a line space between concatenated texts
        if grouped_texts[section_id]:
            grouped_texts[section_id] += "\n\n"  # Add a line space

        grouped_texts[section_id] += text
        grouped_pages[section_id].append(page)
        grouped_ids[section_id].append(textid)

    # Create section lookup
    lookup_dict = {d['section_id']: d['section_name'] for d in sections}

    # Create chunks data for every section, even if there's no text
    for section in sections:
        section_id = section['section_id']
        section_name = lookup_dict.get(section_id, None)
        concatenated_text = grouped_texts.get(section_id, "")

        if len(concatenated_text) > MAX_TEXT_CHAR:
            # Split the concatenated text into semantic chunks
            split_chunks = text_splitter.split_text(concatenated_text)
            for chunk in split_chunks:
                tmp_dict = {
                    'text_chunk': chunk,
                    'section_name': section_name,
                    'text_id': grouped_ids.get(section_id, []),
                    'pages': list(set(grouped_pages.get(section_id, [])))
                }
                chunks_data.append(tmp_dict)
        else:
            tmp_dict = {
                'text_chunk': concatenated_text,
                'section_name': section_name,
                'text_id': grouped_ids.get(section_id, []),
                'pages': list(set(grouped_pages.get(section_id, [])))
            }
            chunks_data.append(tmp_dict)

    return chunks_data

# Process exception sections
def process_exceptiontext(text_chunks, exception_section):
    """
    Processes exception sections and combines their text chunks.

    Args:
        text_chunks (list): A list of dictionaries representing individual text chunks.
        exception_section (list): A list of dictionaries representing flagged exception sections.

    Returns:
        list: A list of dictionaries containing concatenated text for each exception section.
    """
    exception_chunks = []
    for section in exception_section:
        grouped_texts = defaultdict(str)
        sectionid = section['section_id']
        name = section['section_name']
        for entry in text_chunks:
            if entry['section_id'] == sectionid:
                grouped_texts[sectionid] += entry['Text']
        for section_id, concatenated_text in grouped_texts.items():
            tmp_dict = {'Section Name': name, 'Text': concatenated_text}
            exception_chunks.append(tmp_dict)

    return exception_chunks

# Function to call a specific function
def call_function(func, *args):
    """
    Calls a specified function with given arguments.

    Args:
        func (callable): The function to call.
        *args: Arguments to pass to the function.

    Returns:
        Any: The result of the function call.
    """

    return func(*args)

# Function to combine texts
def combine_texts(splittext_func, processextract_func, processexceptiontext_func, inputs_elements_df, specialchar_replacements):
    """
    Combines texts by replacing special characters, splitting elements, processing exceptions, and creating chunks.

    Args:
        splittext_func (callable): The function used to split text elements.
        processextract_func (callable): The function used to process and combine extracted text chunks.
        processexceptiontext_func (callable): The function used to process exception texts.
        inputs_elements_df (pd.DataFrame): The input DataFrame containing text elements.
        specialchar_replacements (dict): A dictionary of regex replacements for special characters.

    Returns:
        tuple: A tuple containing processed data:
            - title, tables, figures, text_chunks, sections, exception_chunks, chunks_data.

    """
    elements_df = inputs_elements_df.replace(specialchar_replacements, regex=True)
    title, tables, figures, text_chunks, sections, exception_section = splittext_func(elements_df)
    exception_chunks = processexceptiontext_func(text_chunks, exception_section)
    chunks_data = processextract_func(text_chunks, sections)
    return title, tables, figures, text_chunks, sections, exception_chunks, chunks_data

# Function to output metadata in final output
def save_metadata(service, parent_folder_id, tables, figures, text_chunks, sections, exception_chunks, chunks_data):
    """
    Saves metadata objects to Google Drive as serialized pickle files.

    Args:
        service (googleapiclient.discovery.Resource): The authenticated Google Drive API service.
        parent_folder_id (str): The ID of the Google Drive folder where metadata files will be uploaded.
        tables (list): List of table metadata.
        figures (list): List of figure metadata.
        text_chunks (list): List of text chunks.
        sections (list): List of sections.
        exception_chunks (list): List of exception section data.
        chunks_data (list): List of final processed text chunks.

    Returns:
        None
    """
    # Pickle the data
    data_dicts = {
        'Tables.pkl': tables,
        'Figures.pkl': figures,
        'Exception_chunks.pkl': exception_chunks,
        'Text_chunks.pkl': text_chunks,
        'Sections.pkl': sections,
        'Final_chunks.pkl': chunks_data
    }
    
    for file_name, data in data_dicts.items():
        file_data = pickle.dumps(data)
        upload_file(service, file_name, 'application/octet-stream', file_data, parent_folder_id)

# Function to save final output
def save_output(service, parent_folder_id, title, chunks_data, final_output):
    """
    Saves processed data to an Excel file and uploads it to Google Drive.

    Args:
        service (googleapiclient.discovery.Resource): The authenticated Google Drive API service.
        parent_folder_id (str): The ID of the Google Drive folder where the Excel file will be uploaded.
        title (list): The extracted title metadata.
        chunks_data (list): The processed chunks of text data.
        final_output (str): The name of the output Excel file.

    Returns:
        None
    """
    # Create DataFrame from chunks_data
    chunks_df = pd.DataFrame(chunks_data)
    chunks_df['Title'] = title[0]['title_name']
    
    # Ensure all sections are present in the DataFrame
    chunks_df = chunks_df[~chunks_df['section_name'].isna()]

    # Ensure even empty 'text_chunk' sections are included
    # Fill missing values in 'text_chunk' column with empty strings
    chunks_df['text_chunk'] = chunks_df['text_chunk'].fillna('')

    # Save DataFrame to Excel in a BytesIO stream
    xlsx_data = BytesIO()
    with pd.ExcelWriter(xlsx_data, engine='openpyxl') as writer:
        chunks_df.to_excel(writer, index=False, sheet_name='Sheet1')
    xlsx_data.seek(0)

    # Upload Excel to Google Drive
    upload_file(service, final_output, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', xlsx_data.read(), parent_folder_id)

# Function to process JSON file
def process_json_file(json_file, output_folder_id, drive_service, xlsx_file_name):
    """
    Processes a JSON file and uploads related metadata to Google Drive.

    Args:
        json_file (dict): A dictionary containing metadata of the JSON file (e.g., 'id', 'name').
        output_folder_id (str): The ID of the Google Drive folder where processed files will be uploaded.
        drive_service (googleapiclient.discovery.Resource): The authenticated Google Drive API service.
        xlsx_file_name (str): The name of the final Excel file to be uploaded.

    Returns:
        None
    """
    json_file_id = json_file['id']  # Extract the file ID from the dictionary
    json_file_name = json_file['name']

    # Define the path to save the downloaded JSON file locally
    json_path = f"./{json_file_name}"

    # Download the file from Google Drive to the local path
    download_file_from_drive(json_file_id, json_path)

    # Now open and process the downloaded JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)
        # Process your JSON data here...
        elements_df = pd.DataFrame(data['elements'])
        title, tables, figures, text_chunks, sections, exception_chunks, chunks_data = combine_texts(
            split_using_pathheader, process_extract, process_exceptiontext, elements_df, specialchar_replacements
        )
        
        # Prepare metadata files for upload
        metadata_files = {
            'Tables.pkl': tables,
            'Figures.pkl': figures,
            'Exception_chunks.pkl': exception_chunks,
            'Text_chunks.pkl': text_chunks,
            'Sections.pkl': sections,
            'Final_chunks.pkl': chunks_data
        }
        
        for file_name, data in metadata_files.items():
            # Save metadata to BytesIO stream
            file_data = BytesIO()
            pickle.dump(data, file_data)
            file_data.seek(0)  # Move to the start of the BytesIO stream
            
            # Upload file to Google Drive
            upload_file(drive_service, file_name, 'application/octet-stream', file_data.read(), output_folder_id)
        
        # Also save other output if needed
        save_output(service,output_folder_id, title, chunks_data, xlsx_file_name)


# Main function for dynamic processing
def main():
    '''
    Requires user specific inputs.

    The google drive folders of interest are is the "PDF_Extrated data" for root_folder_id and "Processed Data" folder for output_root_id

    TO EDIT:
    1. Replace root_folder_id and output_root_id with the folder id of the google drive folders of interest
    2. Edit the string folder name in line 616 to the name of the PDF you are chunking for
    '''
    # Define the folder IDs
    root_folder_id = '1r38pL-SjbkwYBoK5EF1_Ou4sxb3iw0H7' # Set this to the actual root folder ID (PDF Extracted data folder)
    output_root_id = '1zkLENCBiRboBEF_MBk59fBH5efjVUuvW'  # Set this to the actual output root folder ID (Processed Data folder)

    # Get all subfolders in the root folder
    folders = list_folders_in_folder(root_folder_id)

    # Loop through each folder
    for folder in folders:

        folder_id = folder['id']
        folder_name = folder['name']
        target_file_name = 'structuredData_edited.json'
        files = list_files_in_folder(folder_id)
        xlsx_file_name = folder_name
        
        if folder_name != 'Osteoporosis — identification and management in primary care':
            continue
        
        json_file = None
        for file in files:
            if file['name'] == target_file_name:
                json_file = file
                break

        if not json_file:
            print(f"No {json_file} found in {folder_name} (ID: {folder_id}). Skipping folder.")
            continue

        print(f"Processing {json_file} in folder '{folder_name}' (ID: {folder_id}).")

        output_folder_id = get_or_create_folder_in_drive(output_root_id, folder_name)

        # Proceed with processing
        process_json_file(json_file, output_folder_id, service, xlsx_file_name)
        print(f"Processed {json_file} in folder '{folder_name}'.")
        

# Execute the main function
if __name__ == "__main__":
    main()
