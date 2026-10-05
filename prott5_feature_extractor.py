import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import re
import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm

MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
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
    model = T5EncoderModel.from_pretrained(MODEL_NAME).to(device)
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


def read_hemolysis_csv(csv_path):
    """
    Read a hemolysis CSV file and return the sequence list.
    Expected column: SEQUENCE
    """
    df = pd.read_csv(csv_path)

    if "SEQUENCE" not in df.columns:
        raise ValueError(
            f"'SEQUENCE' column not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    sequences = (
        df["SEQUENCE"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return sequences

def get_hemolysis_sequences( cross_val_csv, independent_csv):
    """
    Read sequences from both cross-validation and independent datasets.
    Does not modify any existing dataset-loading methods.
    """
    cross_val_sequences = read_hemolysis_csv(cross_val_csv)
    independent_sequences = read_hemolysis_csv(independent_csv)

    print(f"Cross-validation sequences : {len(cross_val_sequences)}")
    print(f"Independent-test sequences : {len(independent_sequences)}")

    # Check overlap
    cross_set = set(cross_val_sequences)
    independent_set = set(independent_sequences)

    overlap = cross_set.intersection(independent_set)

    if overlap:
        print(
            f"WARNING: {len(overlap)} sequences occur in both "
            f"cross-validation and independent datasets."
        )
    else:
        print("No sequence overlap detected.")

    # Preserve order while removing duplicates
    all_sequences = list(
        dict.fromkeys(cross_val_sequences + independent_sequences)
    )

    print(f"Total unique sequences       : {len(all_sequences)}")

    return cross_val_sequences, independent_sequences, all_sequences

def preprocess_sequence(sequence):
    """
    Preprocess an amino-acid sequence for ProtT5.

    - Replace uncommon/ambiguous amino acids with X.
    - Insert spaces between residues, as required by ProtT5 tokenizer.
    """
    sequence = sequence.strip().upper()

    # Replace uncommon amino acids with X
    sequence = sequence.replace("U", "X")
    sequence = sequence.replace("Z", "X")
    sequence = sequence.replace("O", "X")
    sequence = sequence.replace("B", "X")

    # ProtT5 expects residues separated by spaces
    sequence = " ".join(list(sequence))

    return sequence

def compute_hemolysis_prott5_embeddings(
    sequences,
    model,
    tokenizer,
    device,
    batch_size=64
):
    """
    Compute ProtT5 residue-level embeddings for hemolysis sequences.

    This is a standalone function and does not modify
    the existing embedding pipeline.
    """

    processed_sequences = [
        preprocess_sequence(seq)
        for seq in sequences
    ]

    embeddings = []

    for start in range(0, len(processed_sequences), batch_size):

        batch_sequences = processed_sequences[
            start:start + batch_size
        ]

        encoded = tokenizer(
            batch_sequences,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        batch_embeddings = outputs.last_hidden_state

        for i in range(len(batch_sequences)):

            # Number of non-padding tokens
            seq_len_with_eos = attention_mask[i].sum().item()

            # ProtT5 has:
            # amino acids + <eos>
            # No <cls> token
            residue_len = seq_len_with_eos - 1

            # Keep only amino-acid residue embeddings
            residue_embedding = (
                batch_embeddings[i, :residue_len, :]
                .detach()
                .cpu()
                .numpy()
            )

            embeddings.append(residue_embedding)

        print(
            f"Processed "
            f"{min(start + batch_size, len(sequences))}"
            f"/{len(sequences)} sequences"
        )

        del input_ids
        del attention_mask
        del outputs
        del batch_embeddings

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return processed_sequences, embeddings

def save_hemolysis_prott5_embeddings(
    cross_val_csv,
    independent_csv,
    save_path="./data/hemolysis_prott5.npz",
    batch_size=64
):
    """
    Generate and save ProtT5 embeddings for the hemolysis
    cross-validation and independent datasets.

    Existing embedding methods are untouched.
    """

    # --------------------------------------------------
    # Instantiate device, model and tokenizer
    # --------------------------------------------------
    device = get_device()
    model, tokenizer = get_model_and_tokenizer(device)

    # --------------------------------------------------
    # Read both hemolysis datasets
    # --------------------------------------------------
    (
        cross_val_sequences,
        independent_sequences,
        all_sequences
    ) = get_hemolysis_sequences(
        cross_val_csv,
        independent_csv
    )

    print("\nComputing ProtT5 embeddings...")

    # --------------------------------------------------
    # Compute embeddings
    # --------------------------------------------------
    processed_sequences, embeddings = (
        compute_hemolysis_prott5_embeddings(
            sequences=all_sequences,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size
        )
    )

    # --------------------------------------------------
    # Save embeddings
    # --------------------------------------------------
    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        save_path,
        original_sequences=np.array(
            all_sequences,
            dtype=object
        ),
        processed_sequences=np.array(
            processed_sequences,
            dtype=object
        ),
        embeddings=np.array(
            embeddings,
            dtype=object
        )
    )

    print("\n========================================")
    print("Hemolysis ProtT5 embedding generation")
    print("========================================")
    print(f"Cross-validation sequences : {len(cross_val_sequences)}")
    print(f"Independent sequences     : {len(independent_sequences)}")
    print(f"Total unique sequences    : {len(all_sequences)}")
    print(f"Total embeddings          : {len(embeddings)}")
    print(f"First embedding shape     : {embeddings[0].shape}")
    print(f"Saved to                  : {save_path}")
    print("========================================")



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

    save_embeddings(original_sequences, processed_sequences, all_embeddings, save_path)

def load_embeddings(filepath):
    data = np.load(filepath, allow_pickle=True)
    original_seqs = data["original_sequences"]
    processed_seqs = data["processed_sequences"]
    embeddings = data["embeddings"]

    print(f"Loaded {len(original_seqs)} sequences")
    print(f"Embeddings array shape: {embeddings.shape}")

    return original_seqs, processed_seqs, embeddings

# ------------------------
# Entry point
# ------------------------
if __name__ == "__main__0":
    #Ecoli data
    seq_input_path = "./data/ecoli_clustered_dataset_splits.csv"
    save_path = f"./data/ecoli_prott5.npz"


    compute_save_embeddings(seq_input_path, save_path)

 
    original_seqs, processed_seqs, embeddings = load_embeddings(save_path)
    print(
        f"First sequence length: {len(original_seqs[0])}, "
        f"embedding shape: {embeddings[0].shape}"
    )

if __name__ == "__main__":
    CROSS_VAL_HEMOLYSIS = "./data/hemolysis/cross_val_dataset.csv"
    INDEPENDENT_HEMOLYSIS = "./data/hemolysis/independent_dataset.csv"

    save_hemolysis_prott5_embeddings(
        cross_val_csv=CROSS_VAL_HEMOLYSIS,
        independent_csv=INDEPENDENT_HEMOLYSIS,
        save_path="./data/hemolysis/hemolysis_prott5.npz",
        batch_size=64
    )
