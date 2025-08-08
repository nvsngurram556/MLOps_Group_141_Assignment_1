from google.oauth2 import service_account
from googleapiclient.discovery import build

SERVICE_ACCOUNT_FILE = '/Users/nagavenkatasatyanarayanagurram/.config/dvc/mlopsassignment-468409-8e0ba2dfa401.json'

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=['https://www.googleapis.com/auth/drive'])

service = build('drive', 'v3', credentials=creds)

# List first 10 files
results = service.files().list(pageSize=10, fields="files(id, name)").execute()
items = results.get('files', [])

print("Files in Drive:")
for item in items:
    print(f"{item['name']} ({item['id']})")