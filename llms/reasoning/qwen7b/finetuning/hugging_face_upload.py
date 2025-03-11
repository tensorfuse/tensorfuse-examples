from huggingface_hub import create_repo, upload_folder
hugging_face_org_name = "gane5hvarma"
def upload_lora(folder_path: str, model_name: str):
    # Create a new repository in hugging face
    create_repo(model_name, private=True, exist_ok=True)
    # upload the lora adapter to hugging face
    upload_folder(
        repo_id=f"{hugging_face_org_name}/{model_name}",
        folder_path=folder_path,
        commit_message="unsloth grpo lora"
    )