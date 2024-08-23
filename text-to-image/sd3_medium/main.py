import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusion3Pipeline
import io

app = FastAPI()

model_dir = "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=torch.float16).to(device)


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

# an inference endpoint for image generation
@app.post("/generate")
async def generate_image(data: dict):
    text = data.get("text")
    if not text:
        return {"error": "text field is required"}
    prompt = text

    image = pipe(
    prompt,
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
    ).images[0]

    # Convert the image to a byte stream
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type='image/png')

