from datetime import datetime
import time

from google import genai
from google.genai.types import CreateBatchJobConfig
import pandas as pd

PROJECT_ID = "gsoc25-deepmind"
LOCATION = "us-central1"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
MODEL_ID = "gemini-1.5-flash-001"  # @param {type:"string", isTemplate: true}

INPUT_DATA = "gs://cloud-samples-data/generative-ai/batch/batch_requests_for_multimodal_input_2.jsonl"  # @param {type:"string"}

BUCKET_URI = "gs://gsoc25-deepmind/deepmind-test"  # @param {type:"string"}

gcs_batch_job = client.batches.create(
    model=MODEL_ID,
    src=INPUT_DATA,
    config=CreateBatchJobConfig(dest=BUCKET_URI),
)
# print(gcs_batch_job.name)

gcs_batch_job = client.batches.get(name=gcs_batch_job.name)
# print(gcs_batch_job)

for job in client.batches.list():
    print(job.name, job.create_time, job.state)

# Refresh the job until complete
while gcs_batch_job.state == "JOB_STATE_RUNNING":
    time.sleep(5)
    gcs_batch_job = client.batches.get(name=gcs_batch_job.name)

# Check if the job succeeds
if gcs_batch_job.state == "JOB_STATE_SUCCEEDED":
    print("Job succeeded!")
else:
    print(f"Job: {gcs_batch_job}")
    print(f"Job failed: {gcs_batch_job.error}")


# Loading cloud storage output into a DataFrame

import fsspec
import pandas as pd

fs = fsspec.filesystem("gcs")

file_paths = fs.glob(f"{gcs_batch_job.dest.gcs_uri}/*/predictions.jsonl")

if gcs_batch_job.state == "JOB_STATE_SUCCEEDED":
    # Load the JSONL file into a DataFrame
    df = pd.read_json(f"gs://{file_paths[0]}", lines=True)

    df = df.join(pd.json_normalize(df["response"], "candidates"))
    print(df.to_string())
