import os
import pandas as pd
from google.cloud import storage

#環境変数
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../auth/credential.json"

def load_data_from_gcs(bucket_name="pj_horidasimono", prefix="dataset/train/ElectricalAppliance"):
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    df = pd.DataFrame()
    for blob in blobs:
        bucket = client.get_bucket(bucket_name)
        r = storage.Blob(blob.name, bucket)
        content = r.download_as_string()
        df = df.append(pd.read_json(content))
        print(f"read file {blob.name}...")

    df = df.drop_duplicates(subset="url")
    df = df.reset_index(drop=True)
    return df