import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

app = FastAPI()
device = 0 if torch.cuda.is_available() else -1

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

class AppInput(BaseModel):
    text: str

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

# endpoint for text to speech
@app.post("/tts")
async def generate_text(input: AppInput):
    inputs = processor(text=input.text, return_tensors="pt")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    return {"speech": speech.numpy().tolist()}

