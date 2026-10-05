import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import data_util


# ============================================================
# Configuration
# ============================================================

# ESM2 model
#
# t33_650M -> 1280-dimensional residue embeddings
# t30_150M -> 640-dimensional residue embeddings
# t6_8M   -> 320-dimensional residue embeddings
#
# Use t33_650M because the fusion model expects ESM2_DIM = 1280.
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

BATCH_SIZE = 16
# 650M model is relatively large.
# Increase to 32/64 if GPU memory allows.


# ============================================================
# Device
# ============================================================

def get_device():
    """
    Select CUDA if available, otherwise CPU.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    return device


# ============================================================
# Load ESM2 model and tokenizer
# ============================================================

def get_model_and_tokenizer(device):
    """
    Load ESM2 tokenizer and pretrained model.
    """

    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    print(
        "ESM2 hidden dimension:",
        model.config.hidden_size
    )

    return model, tokenizer


# ============================================================
# Preprocess sequences
# ============================================================

def preprocess_peptides(sequence_examples):
    """
    Replace rare/non-standard amino acids:

        U -> X
        Z -> X
        O -> X
        B -> X

    The replacement preserves sequence length.
    """

    rare_pattern = r"[UZOB]"

    processed_sequences = []
    rare_total = 0

    for sequence in sequence_examples:

        sequence = str(sequence).strip().upper()

        rare_count = len(
            re.findall(
                rare_pattern,
                sequence
            )
        )

        rare_total += rare_count

        processed_sequence = re.sub(
            rare_pattern,
            "X",
            sequence
        )

        processed_sequences.append(
            processed_sequence
        )

    print(
        f"Total rare AA replaced: {rare_total}"
    )

    return processed_sequences, rare_total


# ============================================================
# Read sequences
# ============================================================

def read_sequences(csv_file):
    """
    Read peptide sequences from CSV.
    """

    df = pd.read_csv(csv_file)

    if "sequence" not in df.columns:
        raise ValueError(
            f"'sequence' column not found in {csv_file}"
        )

    sequences = (
        df["sequence"]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    print(
        f"Loaded {len(sequences)} sequences from CSV."
    )

    return sequences


# ============================================================
# Tokenize batch
# ============================================================

def tokenize_batch(
    sequences,
    tokenizer,
    device
):
    """
    Tokenize a batch of peptide sequences.

    We do not use truncation because AMP sequences should
    remain intact. Truncation could destroy the residue-to-
    embedding correspondence.
    """

    encoding = tokenizer(
        sequences,
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_special_tokens_mask=True
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    special_tokens_mask = encoding[
        "special_tokens_mask"
    ].to(device)

    return (
        input_ids,
        attention_mask,
        special_tokens_mask
    )


# ============================================================
# Compute residue-level embeddings
# ============================================================

def compute_residue_embeddings(
    input_ids,
    attention_mask,
    special_tokens_mask,
    model,
    sequences
):
    """
    Extract one embedding for each amino-acid residue.

    The tokenizer may add special tokens such as BOS/EOS.
    Therefore, we identify residue positions using:

        attention_mask == 1
        AND
        special_tokens_mask == 0

    This avoids assuming that residue positions are always
    exactly [1:-1].

    Parameters
    ----------
    input_ids : torch.Tensor
        Shape [B, token_length]

    attention_mask : torch.Tensor
        Shape [B, token_length]

    special_tokens_mask : torch.Tensor
        Shape [B, token_length]

    model : ESM2 model

    sequences : list[str]
        Original processed peptide sequences.

    Returns
    -------
    list[np.ndarray]
        Each array has shape:

            [sequence_length, hidden_dimension]
    """

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

    # Move to CPU once
    hidden_states = hidden_states.cpu()

    attention_mask = attention_mask.cpu()
    special_tokens_mask = special_tokens_mask.cpu()

    embeddings = []

    for i, sequence in enumerate(sequences):

        expected_length = len(sequence)

        # Valid residue positions:
        #
        # 1. Not padding
        # 2. Not a special token
        #
        residue_mask = (
            attention_mask[i].bool()
            & ~special_tokens_mask[i].bool()
        )

        residue_embedding = hidden_states[
            i,
            residue_mask,
            :
        ].numpy()

        actual_length = residue_embedding.shape[0]

        if actual_length != expected_length:

            raise RuntimeError(
                "\nESM2 residue length mismatch\n"
                f"Sequence: {sequence}\n"
                f"Expected length: {expected_length}\n"
                f"Extracted length: {actual_length}\n"
                f"Token IDs: "
                f"{input_ids[i].cpu().tolist()}\n"
                f"Attention mask: "
                f"{attention_mask[i].tolist()}\n"
                f"Special-token mask: "
                f"{special_tokens_mask[i].tolist()}"
            )

        embeddings.append(
            residue_embedding
        )

    return embeddings


# ============================================================
# Main embedding pipeline
# ============================================================

def main(
    csv_path,
    save_path
):

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, tokenizer = get_model_and_tokenizer(
        device
    )

    # --------------------------------------------------------
    # Read sequences
    # --------------------------------------------------------

    original_sequences = read_sequences(
        csv_path
    )

    # --------------------------------------------------------
    # Preprocess sequences
    # --------------------------------------------------------

    processed_sequences, rare_count = (
        preprocess_peptides(
            original_sequences
        )
    )

    print(
        "Rare AA count:",
        rare_count
    )

    # --------------------------------------------------------
    # Verify model dimensionality
    # --------------------------------------------------------

    embedding_dim = model.config.hidden_size

    print(
        "ESM2 embedding dimension:",
        embedding_dim
    )

    if MODEL_NAME == "facebook/esm2_t33_650M_UR50D":

        if embedding_dim != 1280:

            raise RuntimeError(
                "Expected 1280-dimensional embeddings "
                f"but model returned {embedding_dim}."
            )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    all_embeddings = []

    num_sequences = len(
        processed_sequences
    )

    for batch_start in tqdm(
        range(
            0,
            num_sequences,
            BATCH_SIZE
        ),
        desc="Embedding batches"
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            num_sequences
        )

        batch_sequences = (
            processed_sequences[
                batch_start:batch_end
            ]
        )

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        (
            input_ids,
            attention_mask,
            special_tokens_mask
        ) = tokenize_batch(
            batch_sequences,
            tokenizer,
            device
        )

        # ----------------------------------------------------
        # ESM2 forward pass
        # ----------------------------------------------------

        batch_embeddings = (
            compute_residue_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                model=model,
                sequences=batch_sequences
            )
        )

        # ----------------------------------------------------
        # Store embeddings
        # ----------------------------------------------------

        all_embeddings.extend(
            batch_embeddings
        )

        # ----------------------------------------------------
        # Free GPU memory
        # ----------------------------------------------------

        del (
            input_ids,
            attention_mask,
            special_tokens_mask,
            batch_embeddings
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Final checks
    # ========================================================

    print(
        f"\nTotal sequences embedded: "
        f"{len(all_embeddings)}"
    )

    if len(all_embeddings) != len(
        processed_sequences
    ):
        raise RuntimeError(
            "Number of embeddings does not match "
            "number of sequences."
        )

    # Check every sequence
    for sequence, embedding in zip(
        processed_sequences,
        all_embeddings
    ):

        expected_shape = (
            len(sequence),
            embedding_dim
        )

        if embedding.shape != expected_shape:

            raise RuntimeError(
                f"Invalid embedding shape for "
                f"{sequence}: "
                f"expected {expected_shape}, "
                f"got {embedding.shape}"
            )

    print(
        "First sequence:",
        processed_sequences[0]
    )

    print(
        "First embedding shape:",
        all_embeddings[0].shape
    )

    print(
        "Last embedding shape:",
        all_embeddings[-1].shape
    )

    # ========================================================
    # Save embeddings
    # ========================================================

    print(
        f"\nSaving embeddings to:\n{save_path}"
    )

    data_util.save_embeddings(
        processed_sequences,
        all_embeddings,
        save_path
    )

    print("Embedding extraction completed.")

    # ========================================================
    # Reload and verify
    # ========================================================

    embed_dict = data_util.load_embeddings(
        save_path
    )

    print(
        f"Reloaded embeddings: "
        f"{len(embed_dict)} sequences"
    )

    # Verify one example
    first_sequence = processed_sequences[0]

    if first_sequence in embed_dict:

        print(
            "Reloaded first embedding shape:",
            embed_dict[first_sequence].shape
        )

    else:

        print(
            "Warning: first sequence was not found "
            "after reloading."
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    seq_input_path = (
        "./data/ecoli_five_fold_dataset.csv"
    )

    save_path = (
        "./data/esm2/"
        f"{Path(MODEL_NAME).name}_ecoli.npz"
    )

    main(
        csv_path=seq_input_path,
        save_path=save_path
    )