from fastapi import FastAPI
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from  google.cloud import storage
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return{"London":"keywords_of_Richard"}

bucket_name = "airbnbadvice"
source_blob_name = "data"
destination_file_name  ="description_london.csv"

# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     """Downloads a blob from the bucket."""
#     # The ID of your GCS bucket
#     # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

#     storage_client = storage.Client()

#     bucket = storage_client.bucket(bucket_name)

#     # Construct a client side representation of a blob.
#     # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
#     # any content from Google Cloud Storage. As we don't need additional data,
#     # using `Bucket.blob` is preferred here.
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)

#     print(
#         "Downloaded storage object {} from bucket {} to local file {}.".format(
#             source_blob_name, bucket_name, destination_file_name
#         )
#     )

# download_blob(bucket_name, source_blob_name, destination_file_name)
BUCKET_NAME = "airbnbadvice"

def download_csv(model_directory="data", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)
    storage_location = 'airbnbadvice/data/'
    blob = client.blob(storage_location)
    blob.download_to_filename('description_london.csv')
    data = pd.read_csv('description_london.csv')
    print("good, it ran till here")
    return data

download_csv( bucket = BUCKET_NAME)

if __name__ == "__main__":
    index()
    



