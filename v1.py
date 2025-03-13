import time
import os
import pathlib
import fsspec
import pandas as pd
from google import genai
from google.genai import types
from google.genai.types import CreateBatchJobConfig
from google.cloud import storage
from dotenv import load_dotenv
import json

# Load API key from environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
MODEL_ID = "gemini-1.5-flash-001"


def createCache():
    """Creates a cache with a video file and returns it"""

    client = genai.Client(api_key=API_KEY)

    path_to_video_file = pathlib.Path("raspberry-pi.mp4")

    if not path_to_video_file.exists():
        print("Video file not found.")
        return None

    video_file = client.files.upload(file=path_to_video_file)

    while video_file.state.name == "PROCESSING":
        print("Waiting for video to be processed...")
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)

    print(f"Video processing complete: {video_file.uri}")

    cache = client.caches.create(
        model=MODEL_ID,
        config=types.CreateCachedContentConfig(
            display_name="Raspberry Pi Video",
            system_instruction="You are an expert in Raspberry Pi, and your job is to answer questions based on the provided video.",
            contents=[video_file],
            ttl="300s",
        ),
    )

    print(f"Cache created: {cache.name}")
    return cache


def uploadToGcs(input_json_lines):
    """Uploads input json lines to GCS and returns the GCS URI."""

    BUCKET_NAME = os.getenv("BUCKET_NAME")
    LOCAL_FILE_PATH = "input.jsonl"
    GCS_FILE_PATH = "batch_requests_input/input.jsonl"

    with open(LOCAL_FILE_PATH, "w") as f:
        for line in input_json_lines:
            json.dump(line, f)
            f.write("\n")

    storage_client = storage.Client(project="gsoc25-deepmind")
    bucket = storage_client.bucket(BUCKET_NAME)

    print(f"Uploading file to {GCS_FILE_PATH}...")

    blob = bucket.blob(GCS_FILE_PATH)
    blob.upload_from_filename(LOCAL_FILE_PATH)

    gcs_uri = f"gs://{BUCKET_NAME}/{GCS_FILE_PATH}"
    print(f"File uploaded successfully to {gcs_uri}")

    return gcs_uri


def getInputJsonLines(cache):
    """Returns some hardcoded input json lines for the batch job."""
    input_json_lines = [
        {
            "request": {
                "contents": [
                    {"role": "user", "parts": [{"text": "What is a turing machine?"}]}
                ],
                "generationConfig": {"temperature": 0.4},
                "tools": [{"cachedContent": cache.name}],
            }
        },
        {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": "What is the main topic of the video that is in the cache context given to you?"
                            }
                        ],
                    }
                ],
                "generationConfig": {"temperature": 0.4},
                "tools": [{"cachedContent": cache.name}],
            }
        },
    ]

    return input_json_lines


def displayData(gcs_batch_job):
    """Displays the output data from the batch job."""
    
    # Loading cloud storage output into a DataFrame
    fs = fsspec.filesystem("gcs")

    file_paths = fs.glob(f"{gcs_batch_job.dest.gcs_uri}/*/predictions.jsonl")

    if gcs_batch_job.state == "JOB_STATE_SUCCEEDED":
        # Load the JSONL file into a DataFrame
        df = pd.read_json(f"gs://{file_paths[0]}", lines=True)

        df = df.join(pd.json_normalize(df["response"], "candidates"))
        print(df.to_string())


def runBatchJob(gcs_uri):
    """Runs a batch job using the cache and input json lines."""

    PROJECT_ID = os.getenv("PROJECT_ID")
    BUCKET_URI = os.getenv("BUCKET_URI")
    LOCATION = "us-central1"

    INPUT_DATA = gcs_uri

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    batch_job = client.batches.create(
        model=MODEL_ID,
        src=INPUT_DATA,
        config=CreateBatchJobConfig(
            dest=BUCKET_URI,
        ),
    )

    gcs_batch_job = client.batches.get(name=batch_job.name)

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

    displayData(gcs_batch_job)


def main():
    cache = createCache()
    if cache is None:
        raise SystemExit

    input_json_lines = getInputJsonLines(cache)
    gcs_uri = uploadToGcs(input_json_lines)
    runBatchJob(gcs_uri)

if __name__ == "__main__":
    main()