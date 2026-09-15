import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import re
import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm

import data_util

# MODEL_NAME = "Rostlab/prot_t5_xl_uniref50" #1024
MODEL_NAME = "Rostlab/ProstT5" #Structure Aware embedding
BATCH_SIZE = 64  # adjust based on GPU memory


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
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
    model.eval()
    return model, tokenizer


# ------------------------
# Preprocess sequences
# ------------------------
def preprocess_peptides(sequence_examples):
    """
    Replace rare amino acids (U, Z, O, B) with 'X' and insert spaces.
    """
    rare_pattern = r"[UZOB]"
    processed = []
    rare_total = 0

    for seq in sequence_examples:
        rare_count = len(re.findall(rare_pattern, seq))
        rare_total += rare_count
        seq_proc = " ".join(re.sub(rare_pattern, "X", seq))
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
    return encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


# ------------------------
# Compute residue-level embeddings
# ------------------------
def compute_residue_embeddings_error(input_ids, attention_mask, model):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state  # [B, L, 1024]  #

    # remove <cls> and <eos>
    emb = emb[:, 1:-1, :] #Error ProtT5 does not have cls, only have eos at the end
    att = attention_mask[:, 1:-1]

    return emb.detach().cpu().numpy(), att.detach().cpu().numpy()

def compute_residue_embeddings(input_ids, attention_mask, model):
    """
    Extract residue-level ProtT5 embeddings.

    ProtT5 does NOT use a <cls> token.
    The protein sequence is followed by an <eos> token, and
    batches may contain padding.

    Returns:
        residue_embeddings: list of arrays, each with shape (L, 1024)
        residue_masks:      list of arrays, each with shape (L,)
    """

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [B, token_length, 1024]
        emb = outputs.last_hidden_state

    residue_embeddings = []
    residue_masks = []

    for i in range(input_ids.size(0)):

        # Number of non-padding tokens
        seq_len_with_eos = attention_mask[i].sum().item()

        # ProtT5 has:
        #   amino acids + <eos>
        #
        # Remove only the final <eos>.
        residue_len = seq_len_with_eos - 1

        # Keep only amino-acid residue embeddings
        residue_emb = emb[i, :residue_len, :]

        # Corresponding residue mask
        residue_att = attention_mask[i, :residue_len]

        residue_embeddings.append(
            residue_emb.detach().cpu().numpy()
        )

        residue_masks.append(
            residue_att.detach().cpu().numpy()
        )

    return residue_embeddings, residue_masks


# ------------------------
# Save embeddings
# ------------------------
def save_embeddings(original_seqs, processed_seqs, embeddings, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        filepath,
        original_sequences=np.array(original_seqs, dtype=object),
        processed_sequences=np.array(processed_seqs, dtype=object),
        embeddings=np.array(embeddings, dtype=object)
    )
    print(f"Saved embeddings to {filepath}")



# ------------------------
# Main pipeline
# ------------------------
def compute_save_embeddings(csv_path, save_path):
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
        batch_seqs = processed_sequences[batch_start:batch_start + BATCH_SIZE]
        input_ids, attention_mask = tokenize_batch(batch_seqs, tokenizer, device)

        batch_emb, batch_att = compute_residue_embeddings(
            input_ids, attention_mask, model
        )

        for emb in batch_emb:
            all_embeddings.append(emb)

        del input_ids, attention_mask, batch_emb, batch_att
        torch.cuda.empty_cache()

    print(f"Total sequences embedded: {len(all_embeddings)}")
    print(f"First embedding shape: {all_embeddings[0].shape}")

    data_util.save_embeddings(processed_sequences, all_embeddings, save_path)


# ------------------------
# Entry point
# ------------------------
if __name__ == "__main__":
    #Ecoli data
    seq_input_path = "./data/ecoli_five_fold_dataset.csv"
    save_path = f"./data/prott5/{Path(MODEL_NAME).name}_ecoli.npz"

    #Staphylococcus aureus data
    # seq_input_path = "/data/prem001/PGAT-ABPp/code/data/s_aureus_pipcleaned.csv"
    # save_path = f"/data/prem001/PGAT-ABPp/code/prott5/{Path(MODEL_NAME).name}/prott5_s_aureus_residue_level.npz"

    #Pseudomonas aeruginosa data
    # seq_input_path = "./data/p_aeruginosa_dataset.csv"
    # save_path = f"./data/prott5_p_aeruginosa_residue_level.npz"

    compute_save_embeddings(seq_input_path, save_path)

    #Load embeddings
    embedding_dict = data_util.load_embeddings(save_path)