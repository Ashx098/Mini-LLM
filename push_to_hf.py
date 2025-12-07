import os
from huggingface_hub import HfApi, create_repo, upload_folder

# Configuration
HF_USERNAME = "Ashx098"
REPO_NAME = "Mini-LLM"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
TOKENIZER_DIR = "Tokenizer"
DATA_DIR = "data"

def push_to_hub():
    print(f"🚀 Preparing to push to Hugging Face Hub: {REPO_ID}")
    
    api = HfApi()
    
    # 1. Create Repository (if it doesn't exist)
    try:
        print(f"Creating repository {REPO_ID}...")
        # We create a 'model' repo by default. 
        # If you wanted a dataset repo, you'd change repo_type="dataset"
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print("✅ Repository ready.")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        print("💡 Make sure you are logged in with 'huggingface-cli login' or have set HF_TOKEN env var.")
        return

    # 2. Upload Tokenizer
    # We upload the entire Tokenizer folder to the root/Tokenizer path in the repo
    print(f"\n📤 Uploading Tokenizer directory...")
    try:
        api.upload_folder(
            folder_path=TOKENIZER_DIR,
            repo_id=REPO_ID,
            path_in_repo="Tokenizer",
            repo_type="model",
            ignore_patterns=["__pycache__", "*.pyc"]
        )
        print("✅ Tokenizer uploaded.")
    except Exception as e:
        print(f"❌ Error uploading Tokenizer: {e}")

    # 3. Upload Data
    # We upload the data folder. This includes raw text and binary files.
    # Note: Hugging Face handles large files (LFS) automatically.
    print(f"\n📤 Uploading Data directory...")
    try:
        api.upload_folder(
            folder_path=DATA_DIR,
            repo_id=REPO_ID,
            path_in_repo="data",
            repo_type="model",
            ignore_patterns=["__pycache__", "*.pyc"]
        )
        print("✅ Data uploaded.")
    except Exception as e:
        print(f"❌ Error uploading Data: {e}")

    print(f"\n✨ Done! View your repo at: https://huggingface.co/{REPO_ID}")
    print("\nTo use this in the future:")
    print(f"from transformers import AutoTokenizer")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{REPO_ID}', subfolder='Tokenizer/BPE')")

if __name__ == "__main__":
    push_to_hub()
