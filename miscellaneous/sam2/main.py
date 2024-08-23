import torch
from fastapi import FastAPI, UploadFile, File, Form
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import shutil
import numpy as np
import json
from PIL import Image

app = FastAPI()
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

@app.get("/")
async def root():
    is_cuda_available = torch.cuda.is_available()
    return {
        "message": "Hello World-2",
        "cuda_available": is_cuda_available,
    }

@app.get("/readiness")
async def readiness():
    return {"status": "ready"}

# an inference endpoint for text generation
@app.post("/segment")
async def generate_text(data: str = Form(...), image: UploadFile = File(...)):
    with open(image.filename, 'wb') as buffer:
        shutil.copyfileobj(image.file, buffer)

    data = json.loads(data)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        img = np.array(Image.open(image.filename).convert("RGB"))
        predictor.set_image(img)
        input_point = np.array([data["point"]])
        input_label = np.array(data["label"])
        print(data["point"])
        print(data["label"])
        masks, scores, logits = predictor.predict(point_coords=input_point, 
                                                  point_labels=input_label,
                                                  multimask_output=True,)
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        return {"masks": masks.tolist(), "scores": scores.tolist(), "logits": logits.tolist()}
