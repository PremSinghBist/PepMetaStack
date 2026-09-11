from huggingface_hub import snapshot_download

if __name__ == "__main__":
    MODEL_DOWNLOAD_DIR = "/data/prem001/llms"
    models = [
        "facebook/esm2_t12_35M_UR50D",
        "Rostlab/prot_t5_xl_uniref50",
        "Rostlab/prot_t5_xl_half_uniref50-enc",
    ]
    for repo_id in models:
        snapshot_download(repo_id=repo_id, local_dir=f"{MODEL_DOWNLOAD_DIR}/{repo_id}")
