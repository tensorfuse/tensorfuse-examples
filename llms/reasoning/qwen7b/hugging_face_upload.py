from huggingface_hub import create_repo, upload_folder

def upload_lora(folder_path: str, model_name: str):
    # Create a new repository in hugging face
    create_repo(model_name, private=True, exist_ok=True)
    # upload the lora adapter to hugging face
    upload_folder(
        repo_id=f"gane5hvarma/{model_name}",
        folder_path=folder_path,
        commit_message="unsloth grpo lora"
    )