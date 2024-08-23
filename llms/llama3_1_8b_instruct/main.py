import torch
import transformers
from fastapi import FastAPI
import os

app = FastAPI()

model_dir = "models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_dir,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

@app.get("/")
async def root():
    is_cuda_available = torch.cuda.is_available()
    return {
        "message": "Hello World",
        "cuda_available": is_cuda_available,
    }

@app.get("/readiness")
async def readiness():
    return {"status": "ready"}

# an inference endpoint for text generation
@app.post("/generate")
async def generate_text(data: dict):
    text = data.get("text")
    if not text:
        return {"error": "text field is required"}
    prompt = text
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    outputs = pipeline(
    messages,
    max_new_tokens=256,
    )

    response = outputs[0]["generated_text"][-1]["content"]
    return {"generated_text": response}

