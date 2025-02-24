from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from huggingface_hub import snapshot_download
import time
import os
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

print("\n" + "="*50)
print("Starting model and processor loading...")
loading_start_time = time.time()

# Load processor and model
model_path = "deepseek-ai/Janus-Pro-7B"

# Load processor and model from the downloaded snapshot
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True
)

vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

loading_time = time.time() - loading_start_time
print(f"Model and processor loading completed!")
print(f"Total loading time: {loading_time:.2f} seconds")
print("="*50 + "\n")

class AnswerResponse(BaseModel):
    answer: str

# @app.post("/infer", response_model=AnswerResponse)
# async def infer(
#     question: str = Form(...),
#     image: UploadFile = File(...)
# ):
#     try:
#         # Read image file and prepare it for processing
#         image_data = await image.read()
#         pil_image = load_pil_images([image_data])

#         # Prepare conversation input
#         conversation = [
#             {
#                 "role": "<|User|>",
#                 "content": f"\n{question}",
#                 "images": [pil_image],
#             },
#             {"role": "<|Assistant|>", "content": ""},
#         ]

#         # Prepare inputs for the model
#         prepare_inputs = vl_chat_processor(
#             conversations=conversation, 
#             images=pil_image, 
#             force_batchify=True
#         ).to(vl_gpt.device)

#         # Generate embeddings and run inference
#         inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
#         outputs = vl_gpt.language_model.generate(
#             inputs_embeds=inputs_embeds,
#             attention_mask=prepare_inputs.attention_mask,
#             pad_token_id=tokenizer.eos_token_id,
#             bos_token_id=tokenizer.bos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             max_new_tokens=512,
#             do_sample=False,
#             use_cache=True,
#         )

#         # Decode the output to get the answer
#         answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
#         return AnswerResponse(answer=answer)
    
#     except Exception as e:
#         print(f"Error during inference: {str(e)}")
#         return {"error": str(e)}, 500

@app.post("/infer", response_model=AnswerResponse)
async def infer(
    question: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # Read image file and convert to bytes
        image_data = await image.read()
        
        # Convert image data to list of bytes
        image_bytes = [image_data]  # load_pil_images expects a list of bytes
        pil_image = load_pil_images(image_bytes)[0]  # Get the first image since we only send one

        # Prepare conversation input - Fixed the list syntax
        conversation = [
            {
                "role": "<|User|>",
                "content": f"\n{question}",
                "images": [pil_image]
            },
            {
                "role": "<|Assistant|>", 
                "content": ""
            }
        ]

        # Prepare inputs for the model
        prepare_inputs = vl_chat_processor(
            conversations=conversation, 
            images=[pil_image],  # Make sure images is a list
            force_batchify=True
        ).to(vl_gpt.device)

        # Rest of your code remains the same
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return AnswerResponse(answer=answer)
    
    except Exception as e:
        return AnswerResponse(answer=f"Error during inference: {str(e)}")



@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

