import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import data_util

# MODEL_NAME = "facebook/esm2_t33_650M_UR50D" #1280 Dim
# MODEL_NAME = "facebook/esm2_t30_150M_UR50D" #640 Dim 
MODEL_NAME = "facebook/esm2_t6_8M_UR50D" #320 Dim 
BATCH_SIZE = 64  # ⚠ 650M model is heavy — reduce if OOM


# ------------------------
# Device selection
# ------------------------
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


# ------------------------
# Load model + tokenizer
# ------------------------
def get_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return model, tokenizer


# ------------------------
# Preprocess sequences
# ------------------------
def preprocess_peptides(sequence_examples):
    """
    Replace rare amino acids (U, Z, O, B) with 'X'.
    No spacing needed for ESM.
    """
    rare_pattern = r"[UZOB]"
    processed = []
    rare_total = 0

    for seq in sequence_examples:
        rare_count = len(re.findall(rare_pattern, seq))
        rare_total += rare_count

        seq_proc = re.sub(rare_pattern, "X", seq)
        processed.append(seq_proc)

    print(f"Total rare AA replaced: {rare_total}")
    return processed, rare_total


# ------------------------
# Read sequences from CSV
# ------------------------
def read_sequences(csv_file):
    df = pd.read_csv(csv_file)
    seqs = df["sequence"].astype(str).tolist()
    print(f"Loaded {len(seqs)} sequences from CSV.")
    return seqs


# ------------------------
# Tokenize batch
# ------------------------
def tokenize_batch(seqs, tokenizer, device):
    encoding = tokenizer(
        seqs,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        return_tensors="pt"
    )

    return (
        encoding["input_ids"].to(device),
        encoding["attention_mask"].to(device)
    )


# ------------------------
# Compute residue-level embeddings
# ------------------------
def compute_residue_embeddings(input_ids, attention_mask, model):
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        emb = outputs.last_hidden_state  # [B, L, 1280]

    # Remove <cls> and <eos>
    emb = emb[:, 1:-1, :]
    att = attention_mask[:, 1:-1]

    return emb.detach().cpu().numpy(), att.detach().cpu().numpy()




# ------------------------
# Main pipeline
# ------------------------
def main(csv_path, save_path):
    device = get_device()
    model, tokenizer = get_model_and_tokenizer(device)

    original_sequences = read_sequences(csv_path)
    processed_sequences, rare_count = preprocess_peptides(original_sequences)
    print("Rare AA count:", rare_count)

    all_embeddings = []

    for batch_start in tqdm(
        range(0, len(processed_sequences), BATCH_SIZE),
        desc="Embedding batches"
    ):
        batch_seqs = processed_sequences[
            batch_start:batch_start + BATCH_SIZE
        ]

        input_ids, attention_mask = tokenize_batch(
            batch_seqs, tokenizer, device
        )

        batch_emb, batch_att = compute_residue_embeddings(
            input_ids, attention_mask, model
        )

        for j in range(batch_emb.shape[0]):
            seq_len = int(batch_att[j].sum())
            all_embeddings.append(batch_emb[j, :seq_len, :])

        del input_ids, attention_mask, batch_emb, batch_att
        torch.cuda.empty_cache()

    print(f"Total sequences embedded: {len(all_embeddings)}")
    print(f"First embedding shape: {all_embeddings[0].shape}")

    data_util.save_embeddings(
        processed_sequences,
        all_embeddings,
        save_path
    )


# ------------------------
# Entry point
# ------------------------
if __name__ == "__main__":

    seq_input_path = "./data/ecoli_five_fold_dataset.csv"
    
    save_path = (
        f"./data/esm2/"
        f"{Path(MODEL_NAME).name}_ecoli.npz"
    )

    main(seq_input_path, save_path)

    embed_dict = data_util.load_embeddings(save_path)

    