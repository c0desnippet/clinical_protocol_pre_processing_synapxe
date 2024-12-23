import json
import math
import os
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# Authenticate and build the google Drive API client
'''
Requires user specific inputs

Replace SERVICE_ACCOUNT_FILE with the path to your JSON key file.
'''
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = ''
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=creds)

# Function to list files in shared folder
def list_files_in_folder(folder_id):
    """
    Lists all files in a specified Google Drive folder.

    Args:
        folder_id (str): The ID of the Google Drive folder to list files from.

    Returns:
        list: A list of dictionaries representing the files in the folder, each containing metadata like 'name' and 'id'.
    """
    query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder'"
    results = service.files().list(q=query).execute()
    return results.get('files', [])

# Function to list subfolders in folder
def list_folders_in_folder(folder_id):
    """
    Lists all subfolders in a specified Google Drive folder.

    Args:
        folder_id (str): The ID of the Google Drive folder to list subfolders from.

    Returns:
        list: A list of dictionaries representing the subfolders, each containing metadata like 'name' and 'id'.
    """
    query = f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'"                                                                                                                              
    results = service.files().list(q=query).execute()
    return results.get('files', [])

# Function to download file
def download_file(file_id, file_name):
    """
    Downloads a file from Google Drive using its file ID.

    Args:
        file_id (str): The ID of the file to download.
        file_name (str): The name to save the downloaded file as.

    Returns:
        str: The name of the downloaded file.
    """
    request = service.files().get_media(fileId=file_id)
    with open(file_name, 'wb') as file:
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%")
    return file_name  # Return the name of the downloaded file

# Function to upload file
def upload_file(file_path, folder_id, service):
    """
    Uploads a file to a specified Google Drive folder.

    Args:
        file_path (str): The path of the file to upload.
        folder_id (str): The ID of the Google Drive folder where the file will be uploaded.
        service (googleapiclient.discovery.Resource): The authenticated Google Drive API service.

    Returns:
        None
    """
    file_metadata = {
        'name': file_path.split('/')[-1],  # Name of the file
        'parents': [folder_id]  # Parent folder ID
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Uploaded file with ID: {file.get('id')}")


# Function to remove fields with NaN values
def remove_nan_fields(obj):
    if isinstance(obj, dict):
        return {k: remove_nan_fields(v) for k, v in obj.items() if not (isinstance(v, float) and math.isnan(v))}
    elif isinstance(obj, list):
        return [remove_nan_fields(v) for v in obj]
    return obj

# Function to process the Excel file and update the JSON file
def process_files(excel_file_path, json_file_path, updated_json_file_path):
    """
    Processes an Excel file and a JSON file to update the JSON based on the Excel data.

    Args:
        excel_file_path (str): Path to the Excel file to process.
        json_file_path (str): Path to the JSON file to update.
        updated_json_file_path (str): Path where the updated JSON file will be saved.

    Returns:
        None

    """
    # Load the Excel file
    df = pd.read_excel(excel_file_path)

    # Load JSON file
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)
        elements_df = pd.DataFrame(json_data['elements'])

    # Process each row in the Excel file
    for row_number, row in df.iterrows():
        Files = row['Files']
        ObjectID_file = row['ObjectID_file']
        Summaries = row['Summaries']  # Ensure this column exists in the Excel file
        General_Paths = row['General Paths']
        Specific_Paths = row['Specific Paths']

        print(f'Files col in excel: {Files}')
        print(f'Corresponding Summary in excel: {Summaries}')
        print(f'Corresponding Objectid in excel: {ObjectID_file}')

        for index_number, index in elements_df.iterrows():
            Text = index['Text']
            ObjectID = index['ObjectID']
            filePaths = index['filePaths']
            Path = index['Path']

            if ObjectID == ObjectID_file and isinstance(filePaths, list):
                print(f'Match found in JSON file, objectid in json: {ObjectID_file} and current Text in JSON: {Text}')

                # Update the DataFrame directly using .at or .loc
                elements_df.at[index_number, 'Text'] = Summaries
                new_text = elements_df.at[index_number, 'Text']
                print(f'Text changed in JSON: {new_text}')
                
            elif Path in Specific_Paths and isinstance(filePaths, float):
                
                print(f'Deletion required in JSON file, objectid in json: {ObjectID_file} and current Text in JSON: {Text}')

                # Update the DataFrame directly using .at or .loc
                elements_df.at[index_number, 'Text'] = ""
                new_text = elements_df.at[index_number, 'Text']
                print(f'Text changed in JSON: {new_text}')
                    

    # Convert the updated DataFrame back to a list of dictionaries
    updated_elements = elements_df.to_dict(orient='records')
    json_data['elements'] = updated_elements

    # Remove fields with NaN values
    json_data = remove_nan_fields(json_data)

    # Save the updated JSON as a new file
    with open(updated_json_file_path, 'w') as updated_json_file:
        json.dump(json_data, updated_json_file, indent=4)

    print("JSON file updated successfully.")

    # Get the path to the Downloads folder
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    # Set the CSV output path to the Downloads folder
    csv_output_path = os.path.join(downloads_folder, 'updated_elements.csv')

    # Output the updated elements_df as a CSV file
    elements_df.to_csv(csv_output_path, index=False)
    print(f"Updated elements DataFrame exported to CSV in the Downloads folder: {csv_output_path}")

# Main Application
def main():
    """
    Main function to process files in a specified Google Drive folder and its subfolders.

    Steps:
        - Lists subfolders in the root folder.
        - For each subfolder, downloads necessary files (Excel and JSON).
        - Updates the JSON file based on the Excel file.
        - Uploads the updated JSON file back to Google Drive.

        
    The google drive folder of interest in this script is the "PDF_Extrated data" subfolder under the "Data" folder.

    Requires user specific inputs.

    Replace root_folder_id with the folder id of the google drive folder of interest.
    """
    # Define the folder ID
    root_folder_id = '1r38pL-SjbkwYBoK5EF1_Ou4sxb3iw0H7'

    # List all subfolders in the root folder
    folders = list_folders_in_folder(root_folder_id)

    for folder in folders:
        folder_id = folder['id']
        folder_name = folder['name']

        print(f"Processing folder: {folder_name}")

        # List all files in the folder
        files = list_files_in_folder(folder_id)

        # Check if structuredData_edited.json exists in the folder
        if any(file['name'] == 'structuredData_edited.json' for file in files):
            print(f"Skipping folder {folder_name} as structuredData_edited.json already exists.")
            continue 

        # Variable to hold the path of the JSON file (if found)
        json_file_path = None

        for file in files:
            file_id = file['id']
            file_name = file['name']
            mime_type = file['mimeType']

            if mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                # Download the Excel file
                excel_file_path = download_file(file_id, file_name)
                print(f"Downloaded Excel file: {excel_file_path}")

            elif mime_type == 'application/json' and file_name == 'structuredData.json':
                # Download the JSON file
                json_file_path = download_file(file_id, file_name)
                print(f"Downloaded JSON file: {json_file_path}")

        # Ensure the JSON file is downloaded
        if json_file_path:
            # Define the path for the updated JSON file
            updated_json_file_path = 'structuredData_edited.json'

            # Process the downloaded Excel file and create the updated JSON
            process_files(excel_file_path, json_file_path, updated_json_file_path)

            # Upload the updated JSON file back to Google Drive
            upload_file(updated_json_file_path, folder_id, service)
        else:
            print("No structuredData.json file found in the folder.")

if __name__ == '__main__':
    main()
