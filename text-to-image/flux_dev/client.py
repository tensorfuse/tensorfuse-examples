import requests
import json
from io import BytesIO
from PIL import Image
import numpy as np
import os
deployment_url = "<DEPLOYMENT_URL>"
api_key = os.getenv("API_KEY")
inference_endpoint = f"{deployment_url}/v2/models/flux/versions/1/infer"

request_data = {
    "inputs": [
      {
        "name": "PROMPT",
        "shape": [1],
        "datatype": "BYTES",
        "data": ["Generate a golden retriever with a sunset background"]
      }
    ]
}

headers = {"Content-Type": "application/json", "API_KEY": api_key}

# Send POST request
response = requests.post(inference_endpoint, headers=headers, json=request_data)
if response.status_code != 200:
    print(f"Failed to send request to {inference_endpoint}")
    print(f"Response: {response.text}")
    exit()
response_data = response.json()
image_data = response_data["outputs"][0]["data"]
img_np = np.array(image_data, dtype=np.uint8)
byte_data = img_np.tobytes()
# Wrap the bytes in a BytesIO stream
byte_io = BytesIO(byte_data)

# Save the generated image
generated_image = Image.open(byte_io)
generated_image.save("generated_image.png")