from huggingface_hub import HfApi

# CONFIGURATION
# -----------------------------------------------
YOUR_USERNAME = "ssaha007"
NEW_MODEL_REPO = "cancer-classifier"  # Name for your model storage
MODEL_FILENAME = "Model"         # <--- CHANGE THIS to your exact filename (e.g., pytorch_model.bin)
# -----------------------------------------------

repo_id = f"{YOUR_USERNAME}/{NEW_MODEL_REPO}"
api = HfApi()

print(f"Creating repo: {repo_id}...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

print(f"Uploading {MODEL_FILENAME}...")
api.upload_file(
    path_or_fileobj=MODEL_FILENAME,
    path_in_repo=MODEL_FILENAME,
    repo_id=repo_id,
    repo_type="model"
)
print("✅ Upload Complete!")