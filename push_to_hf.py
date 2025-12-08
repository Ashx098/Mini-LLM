import os
from huggingface_hub import HfApi, create_repo, upload_folder
import plot_logs  # Import plotting script

# Configuration
HF_USERNAME = "Ashx098"
REPO_NAME = "Mini-LLM"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
#TOKENIZER_DIR = "Tokenizer"
#DATA_DIR = "data"
CHECKPOINT_DIR = "out_production"
LOGS_DIR = "logs"
PLOTS_DIR = "plots"
PHASE_FOLDER = "phase-1-pretraining"

def push_to_hub():
    print(f"🚀 Preparing to push to Hugging Face Hub: {REPO_ID}")
    
    api = HfApi()
    
    # 1. Create Repository (if it doesn't exist)
    try:
        print(f"Creating repository {REPO_ID}...")
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print("✅ Repository ready.")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        print("💡 Make sure you are logged in with 'huggingface-cli login' or have set HF_TOKEN env var.")
        return

    # 2. Upload Tokenizer
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

    # 4. Generate Plots
    print(f"\n📊 Generating training plots...")
    try:
        plot_logs.plot_logs(log_dir=LOGS_DIR, output_dir=PLOTS_DIR)
        print("✅ Plots generated.")
    except Exception as e:
        print(f"❌ Error generating plots: {e}")

    # 5. Upload Phase 1 Artifacts (Model, Logs, Plots)
    print(f"\n📤 Uploading Phase 1 Artifacts to '{PHASE_FOLDER}'...")
    
    # Upload Checkpoints
    try:
        print(f"   - Uploading Checkpoints from {CHECKPOINT_DIR}...")
        api.upload_folder(
            folder_path=CHECKPOINT_DIR,
            repo_id=REPO_ID,
            path_in_repo=f"{PHASE_FOLDER}/checkpoints",
            repo_type="model"
        )
        print("   ✅ Checkpoints uploaded.")
    except Exception as e:
        print(f"   ❌ Error uploading checkpoints: {e}")

    # Upload Logs
    try:
        print(f"   - Uploading Logs from {LOGS_DIR}...")
        api.upload_folder(
            folder_path=LOGS_DIR,
            repo_id=REPO_ID,
            path_in_repo=f"{PHASE_FOLDER}/logs",
            repo_type="model"
        )
        print("   ✅ Logs uploaded.")
    except Exception as e:
        print(f"   ❌ Error uploading logs: {e}")

    # Upload Plots
    try:
        print(f"   - Uploading Plots from {PLOTS_DIR}...")
        api.upload_folder(
            folder_path=PLOTS_DIR,
            repo_id=REPO_ID,
            path_in_repo=f"{PHASE_FOLDER}/plots",
            repo_type="model"
        )
        print("   ✅ Plots uploaded.")
    except Exception as e:
        print(f"   ❌ Error uploading plots: {e}")

    print(f"\n✨ Done! View your repo at: https://huggingface.co/{REPO_ID}/tree/main/{PHASE_FOLDER}")

if __name__ == "__main__":
    push_to_hub()
