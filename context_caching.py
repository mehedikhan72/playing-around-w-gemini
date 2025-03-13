from google import genai
import os
import pathlib
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

client = genai.Client(api_key=API_KEY)

path_to_video_file = pathlib.Path("raspberry-pi.mp4")
if not path_to_video_file.exists():
    print("Video file not found.")
    raise SystemExit

# Upload the video using the Files API
video_file = client.files.upload(file=path_to_video_file)

# Wait for the file to finish processing
while video_file.state.name == "PROCESSING":
    print("Waiting for video to be processed.")
    time.sleep(2)
    video_file = client.files.get(name=video_file.name)

print(f"Video processing complete: {video_file.uri}")

# You must use an explicit version suffix. "-flash-001", not just "-flash".
model = "gemini-1.5-flash-001"

# Create a cache with a 5 minute TTL
cache = client.caches.create(
    model=model,
    config=types.CreateCachedContentConfig(
        display_name="Rasbperry Pi",  # used to identify the cache
        system_instruction=(
            "You are an engineer experting in Raspberry Pi and your job is to answer "
            "the user's query based on the video file you have access to."
        ),
        contents=[video_file],
        ttl="300s",
    ),
)

print(cache.name)
print(cache)

# Construct a GenerativeModel which uses the created cache.
response = client.models.generate_content(
    model=model,
    contents=(
        "Whhat is the video about? "
        "How does the video explain the Raspberry Pi? "
        "How does the guy look like in the video?"
    ),
    config=types.GenerateContentConfig(cached_content=cache.name),
)

print(response.usage_metadata)

# The output should look something like this:
#
# prompt_token_count: 696219
# cached_content_token_count: 696190
# candidates_token_count: 214
# total_token_count: 696433

print(response.text)
