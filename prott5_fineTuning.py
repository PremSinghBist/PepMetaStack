import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from prott5_span_corruption import (
    T5SpanCorruptionCollator,
)

# ============================================================
# Configuration
# ============================================================

CSV_PATH = "./data/amp_dataset.csv"

MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"

OUTPUT_DIR = "./model/amp_prott5_partial"

MAX_LENGTH = 100

# Fine-tune last N encoder blocks
NUM_TRAINABLE_LAYERS = 6

# T5 span corruption
NOISE_DENSITY = 0.15
MEAN_NOISE_SPAN_LENGTH = 3

# Training
NUM_EPOCHS = 150

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

# H100
TRAIN_BATCH_SIZE = 512 #2048
EVAL_BATCH_SIZE = 512 #2048
GRADIENT_ACCUMULATION_STEPS = 2

# Early stopping
EARLY_STOPPING_PATIENCE = 15

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# H100 optimization
# ============================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# Dataset
# ============================================================

class AMPDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=256,
    ):

        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        sequence = self.df.iloc[idx]["sequence"]

        # ProtT5 expects amino acids separated by spaces
        sequence = " ".join(sequence)

        encoded = self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": encoded["input_ids"],
        }




# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(
    CSV_PATH
)


# ============================================================
# Validate required columns
# ============================================================

required_columns = [
    "sequence",
    "split",
]

missing = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# Clean sequences
# ============================================================

df["sequence"] = (
    df["sequence"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df = df[
    df["sequence"].str.len() > 0
].copy()


# ============================================================
# Validate split values
# ============================================================

valid_splits = {
    "train",
    "val",
    "test",
}

invalid_splits = (
    set(df["split"].unique())
    - valid_splits
)

if invalid_splits:

    raise ValueError(
        f"Invalid split values: {invalid_splits}"
    )


# ============================================================
# Check exact sequence leakage
# ============================================================

duplicate_count = (
    df["sequence"].duplicated().sum()
)

print(
    f"\nDuplicate sequence rows: "
    f"{duplicate_count}"
)


sequence_split_counts = (
    df.groupby("sequence")["split"]
    .nunique()
)

cross_split_sequences = (
    sequence_split_counts[
        sequence_split_counts > 1
    ]
)

if len(cross_split_sequences) > 0:

    raise ValueError(
        "\nDATA LEAKAGE DETECTED!\n"
        f"{len(cross_split_sequences)} "
        "sequences occur in multiple splits."
    )


# ============================================================
# Remove exact duplicates
# ============================================================

df = df.drop_duplicates(
    subset=["sequence"],
    keep="first",
).copy()


# ============================================================
# Dataset statistics
# ============================================================

print("\nDataset:")

print(
    df["split"]
    .value_counts()
    .sort_index()
)


print("\nSequence length:")

print(
    df["sequence"]
    .str.len()
    .describe()
)


# ============================================================
# Create train / validation / test datasets
# ============================================================

train_df = df[
    df["split"] == "train"
].copy()

val_df = df[
    df["split"] == "val"
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print("\nFinal datasets:")

print(
    f"Train:      {len(train_df)}"
)

print(
    f"Validation: {len(val_df)}"
)

print(
    f"Test:       {len(test_df)}"
)


# ============================================================
# Short sequence statistics
# ============================================================

short_sequences = (
    df["sequence"].str.len() < 5
).sum()

print(
    f"\nSequences shorter than 5 residues: "
    f"{short_sequences}"
)


# ============================================================
# Validate amino acid characters
# ============================================================

valid_amino_acids = set(
    "ACDEFGHIKLMNPQRSTVWY"
)

invalid_sequences = []

for sequence in df["sequence"]:

    invalid_chars = (
        set(sequence)
        - valid_amino_acids
    )

    if invalid_chars:

        invalid_sequences.append(
            (
                sequence,
                invalid_chars,
            )
        )


if invalid_sequences:

    print(
        "\nWARNING:"
        f" {len(invalid_sequences)} sequences "
        "contain non-standard amino-acid characters."
    )

    print(
        "Example:"
        f" {invalid_sequences[:5]}"
    )

    print(
        "\nProtT5 commonly maps uncommon "
        "residues to X. Consider cleaning "
        "these sequences before training."
    )


# ============================================================
# Tokenizer
# ============================================================

tokenizer = T5Tokenizer.from_pretrained(
    MODEL_NAME,
    do_lower_case=False,
)


# ============================================================
# Datasets
# ============================================================

train_dataset = AMPDataset(
    train_df,
    tokenizer,
    MAX_LENGTH,
)

val_dataset = AMPDataset(
    val_df,
    tokenizer,
    MAX_LENGTH,
)


# ============================================================
# Data collator
# ============================================================

data_collator = T5SpanCorruptionCollator(
    tokenizer=tokenizer,
    noise_density=NOISE_DENSITY,
    mean_noise_span_length=MEAN_NOISE_SPAN_LENGTH,
)


# ============================================================
# Load ProtT5
# ============================================================

print(
    "\nLoading ProtT5..."
)

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)


# ============================================================
# Freeze entire model
# ============================================================

for param in model.parameters():

    param.requires_grad = False


# ============================================================
# Fine-tune last N encoder blocks
# ============================================================

encoder_blocks = model.encoder.block

num_layers = len(
    encoder_blocks
)

start_layer = (
    num_layers
    - NUM_TRAINABLE_LAYERS
)

if start_layer < 0:

    raise ValueError(
        "NUM_TRAINABLE_LAYERS is larger "
        "than the number of encoder layers."
    )


print(
    f"\nTotal encoder layers: "
    f"{num_layers}"
)

print(
    f"Trainable encoder layers: "
    f"{start_layer} -> {num_layers - 1}"
)


for layer_idx in range(
    start_layer,
    num_layers,
):

    for param in (
        encoder_blocks[layer_idx]
        .parameters()
    ):

        param.requires_grad = True #Enable grad for last 6 layers


# ============================================================
# Encoder final layer normalization
# ============================================================

for param in (
    model.encoder.final_layer_norm
    .parameters()
):

    param.requires_grad = True


# ============================================================
# Gradient checkpointing
# ============================================================

model.gradient_checkpointing_enable()

# Required because most model parameters
# are frozen while using checkpointing.
model.enable_input_require_grads() #all grad computation wrt to input embeddings

# Required for T5 with gradient checkpointing.
model.config.use_cache = False


# ============================================================
# Parameter statistics
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

trainable_percentage = (
    100.0
    * trainable_params
    / total_params
)


print(
    f"\nTotal parameters: "
    f"{total_params:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_params:,}"
)

print(
    f"Trainable percentage: "
    f"{trainable_percentage:.2f}%"
)


# ============================================================
# Training arguments
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # H100
    # --------------------------------------------------------

    bf16=True,
    tf32=True,

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    per_device_train_batch_size=(
        TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    warmup_ratio=WARMUP_RATIO, 

    lr_scheduler_type="cosine",

    max_grad_norm=1.0,

    optim="adamw_torch_fused",

    # --------------------------------------------------------
    # Training duration
    # --------------------------------------------------------

    num_train_epochs=NUM_EPOCHS,

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="steps",

    eval_steps=50,

    # --------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=50,

    save_total_limit=1,

    save_only_model=True,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=10,

    logging_first_step=True,

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------

    dataloader_num_workers=8,

    dataloader_pin_memory=True,

    # --------------------------------------------------------
    # Miscellaneous
    # --------------------------------------------------------

    report_to="none",

    remove_unused_columns=False,

    seed=SEED,
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    data_collator=data_collator,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=(
                EARLY_STOPPING_PATIENCE
            )
        )
    ],
)


# ============================================================
# Train
# ============================================================

print(
    "\n=========================================="
)

print(
    "Starting AMP-ProtT5 training..."
)

print(
    "==========================================\n"
)

trainer.train()


# ============================================================
# Save best model
# ============================================================

BEST_MODEL_DIR = os.path.join(
    OUTPUT_DIR,
    "best_amp_prott5",
)


trainer.save_model(
    BEST_MODEL_DIR
)

tokenizer.save_pretrained(
    BEST_MODEL_DIR
)


# ============================================================
# Final information
# ============================================================

print(
    "\n=========================================="
)

print(
    "AMP-ProtT5 training completed."
)

print(
    "=========================================="
)

print(
    f"\nBest model saved to:\n"
    f"{BEST_MODEL_DIR}"
)

print(
    "\nTest dataset was NOT used during training "
    "or model selection."
)