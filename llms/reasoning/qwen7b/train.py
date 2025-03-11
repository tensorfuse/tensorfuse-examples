from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
import torch
from reward_functions import xmlcount_reward_func, soft_format_reward_func, strict_format_reward_func, int_reward_func, correctness_reward_func, dataset
from trl import GRPOConfig, GRPOTrainer
from hugging_face_upload import upload_lora
import os 
import json
import wandb
from tensorkube import get_queued_message

message = json.loads(get_queued_message())

PatchFastRL("GRPO", FastLanguageModel)
max_seq_length = message.get('max_seq_len') or 1024 # Can increase for longer reasoning traces
lora_rank = message.get('lora_rank') or 16 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = message.get("model_name") or "Qwen/Qwen2.5-7B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = message.get("gpu_memory_utilization") or 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # lora rank should be in multiples of 2. Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


# wandb config
project_name = message.get("wandb_project_name") or "unsloth"
wandb.init(project=project_name)

# train config
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = message.get("learning_rate") or 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = message.get("lr_scheduler_type") or "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = message.get("max_prompt_length") or 256,
    max_completion_length = message.get("max_completion_length") or 200,
    num_train_epochs = message.get("num_train_epochs") or 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs", # stores the checkpoints in outputs folder
)


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()
folder_name = "lora_adapter"
model.save_lora(folder_name)
print("model trained and saved")
folder_path = os.getcwd() + "/" + folder_name

upload_lora(folder_path, folder_name)