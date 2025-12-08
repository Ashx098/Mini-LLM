import os
from huggingface_hub import HfApi, create_repo, upload_folder
import plot_logs
import convert_to_safetensors # Import conversion script

# Configuration
HF_USERNAME = "Ashx098"
REPO_NAME = "Mini-LLM"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
#TOKENIZER_DIR = "Tokenizer"
#DATA_DIR = "data"
CHECKPOINT_DIR = "out"
SAFETENSORS_DIR = "out_safetensors"
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
        # Continue anyway, repo might exist
    
    # ... (Tokenizer and Data upload skipped for brevity if already done, but keeping structure)
    # For this specific update request, I will keep the existing structure but add the new steps.
    
    # 2. Upload Tokenizer
    if 'TOKENIZER_DIR' in globals() and os.path.exists(TOKENIZER_DIR):
        print(f"\n📤 Uploading Tokenizer directory...")
        try:
            api.upload_folder(folder_path=TOKENIZER_DIR, repo_id=REPO_ID, path_in_repo="Tokenizer", repo_type="model", ignore_patterns=["__pycache__", "*.pyc"])
            print("✅ Tokenizer uploaded.")
        except Exception as e:
            print(f"❌ Error uploading Tokenizer: {e}")
    else:
        print("\n⚠️ Skipping Tokenizer upload (directory not found or variable commented out).")

    # 3. Upload Data
    if 'DATA_DIR' in globals() and os.path.exists(DATA_DIR):
        print(f"\n📤 Uploading Data directory...")
        try:
            api.upload_folder(folder_path=DATA_DIR, repo_id=REPO_ID, path_in_repo="data", repo_type="model", ignore_patterns=["__pycache__", "*.pyc"])
            print("✅ Data uploaded.")
        except Exception as e:
            print(f"❌ Error uploading Data: {e}")
    else:
        print("\n⚠️ Skipping Data upload (directory not found or variable commented out).")

    # 4. Generate Plots
    if os.path.exists(LOGS_DIR):
        print(f"\n📊 Generating training plots...")
        try:
            plot_logs.plot_logs(log_dir=LOGS_DIR, output_dir=PLOTS_DIR)
            print("✅ Plots generated.")
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    else:
        print(f"\n⚠️ Skipping Plot Generation (Logs directory '{LOGS_DIR}' not found).")

    # 5. Convert to Safetensors
    ckpt_path = os.path.join(CHECKPOINT_DIR, "ckpt.pt")
    if os.path.exists(ckpt_path):
        print(f"\n📦 Converting to Safetensors...")
        try:
            convert_to_safetensors.convert_to_safetensors(ckpt_path, SAFETENSORS_DIR)
            print("✅ Conversion successful.")
        except Exception as e:
            print(f"❌ Error converting to safetensors: {e}")
    else:
        print(f"\n⚠️ Skipping Safetensors Conversion (Checkpoint '{ckpt_path}' not found).")

    # 6. Upload Phase 1 Artifacts
    print(f"\n📤 Uploading Phase 1 Artifacts to '{PHASE_FOLDER}'...")
    
    # Upload Checkpoints (PyTorch)
    if os.path.exists(CHECKPOINT_DIR):
        try:
            print(f"   - Uploading PyTorch Checkpoints from {CHECKPOINT_DIR}...")
            api.upload_folder(folder_path=CHECKPOINT_DIR, repo_id=REPO_ID, path_in_repo=f"{PHASE_FOLDER}/checkpoints", repo_type="model")
            print("   ✅ PyTorch Checkpoints uploaded.")
        except Exception as e:
            print(f"   ❌ Error uploading checkpoints: {e}")
    else:
        print(f"   ⚠️ Skipping Checkpoint upload (Directory '{CHECKPOINT_DIR}' not found).")

    # Upload Safetensors
    if os.path.exists(SAFETENSORS_DIR):
        try:
            print(f"   - Uploading Safetensors from {SAFETENSORS_DIR}...")
            api.upload_folder(folder_path=SAFETENSORS_DIR, repo_id=REPO_ID, path_in_repo=f"{PHASE_FOLDER}/safetensors", repo_type="model")
            print("   ✅ Safetensors uploaded.")
        except Exception as e:
            print(f"   ❌ Error uploading safetensors: {e}")
    else:
        print(f"   ⚠️ Skipping Safetensors upload (Directory '{SAFETENSORS_DIR}' not found).")

    # Upload Logs
    if os.path.exists(LOGS_DIR):
        try:
            print(f"   - Uploading Logs from {LOGS_DIR}...")
            api.upload_folder(folder_path=LOGS_DIR, repo_id=REPO_ID, path_in_repo=f"{PHASE_FOLDER}/logs", repo_type="model")
            print("   ✅ Logs uploaded.")
        except Exception as e:
            print(f"   ❌ Error uploading logs: {e}")
    else:
        print(f"   ⚠️ Skipping Logs upload (Directory '{LOGS_DIR}' not found).")

    # Upload Plots
    if os.path.exists(PLOTS_DIR):
        try:
            print(f"   - Uploading Plots from {PLOTS_DIR}...")
            api.upload_folder(folder_path=PLOTS_DIR, repo_id=REPO_ID, path_in_repo=f"{PHASE_FOLDER}/plots", repo_type="model")
            print("   ✅ Plots uploaded.")
        except Exception as e:
            print(f"   ❌ Error uploading plots: {e}")
    else:
        print(f"   ⚠️ Skipping Plots upload (Directory '{PLOTS_DIR}' not found).")

    # 7. Upload Root Files (README.md and model_index.json)
    print(f"\n📤 Uploading Root Files...")
    try:
        # Upload HF_README.md as README.md
        if os.path.exists("HF_README.md"):
            api.upload_file(
                path_or_fileobj="HF_README.md",
                path_in_repo="README.md",
                repo_id=REPO_ID,
                repo_type="model"
            )
            print("   ✅ README.md uploaded.")
        
        # Upload model_index.json
        if os.path.exists("model_index.json"):
            api.upload_file(
                path_or_fileobj="model_index.json",
                path_in_repo="model_index.json",
                repo_id=REPO_ID,
                repo_type="model"
            )
            print("   ✅ model_index.json uploaded.")
            
    except Exception as e:
        print(f"   ❌ Error uploading root files: {e}")

    print(f"\n✨ Done! View your repo at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    push_to_hub()
