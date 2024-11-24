from google.cloud import storage
import os


def upload_to_gcs(bucket_name, local_path, gcs_path):
    """Uploads a file or directory to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    if os.path.isdir(local_path):
        for root, _, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_path)
                gcs_file = os.path.join(gcs_path, relative_path)
                blob = bucket.blob(gcs_file)
                blob.upload_from_filename(local_file)
                print(f"Uploaded {local_file} to {gcs_file}")
    else:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to {gcs_path}")
