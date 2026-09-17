import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import data_util

# ============================================================
# Configuration
# ============================================================

PROTT5_FILE = "./data/prott5/ProstT5_ecoli.npz"
ESM2_FILE = "./data/esm2/esm2_t33_650M_UR50D_1280_ecoli.npz"

OUTPUT_DIR = Path("fusion_result/fusion_model_ESM2_ProtT5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_FILE = OUTPUT_DIR / "best_fusion_model.pt"
FUSED_EMBEDDINGS_FILE = OUTPUT_DIR / "fused_embeddings.npy"

SEED = 42
EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 1e-4

# Model dimensions
PROTT5_DIM = 1024
ESM2_DIM = 1280
FUSION_DIM = 1024

NUM_HEADS = 8
NUM_LAYERS = 2
FF_DIM = 2048
DROPOUT = 0.1

# Masked reconstruction
MASK_PROB = 0.15

# Training
# Training
BATCH_SIZE = 512
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5


GRAD_CLIP = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print(f"Device: {DEVICE}")


# ============================================================
# Embedding loading
# ============================================================

def load_embeddings(npz_file):
    """
    Load embeddings from an NPZ file.

    Assumption:
        loading returns a dictionary:

            {
                "SEQUENCE1": array of shape [L, D],
                "SEQUENCE2": array of shape [L, D],
                ...
            }

    Replace the body of this function with your existing
    loading method if necessary.
    """
    import data_util
    embeddings = data_util.load_embeddings(npz_file)
    
    return embeddings


    

def check_embedding_lengths(prott5_embeddings, esm2_embeddings):

    exact_match = 0
    esm2_one_extra = 0
    esm2_other = 0

    examples = []

    for sequence in prott5_embeddings:

        if sequence not in esm2_embeddings:
            continue

        prott5_len = prott5_embeddings[sequence].shape[0]
        esm2_len = esm2_embeddings[sequence].shape[0]
        seq_len = len(sequence)

        if prott5_len == seq_len and esm2_len == seq_len:
            exact_match += 1

        elif esm2_len == seq_len + 1:
            esm2_one_extra += 1

            if len(examples) < 10:
                examples.append(
                    (
                        sequence,
                        seq_len,
                        prott5_len,
                        esm2_len
                    )
                )

        else:
            esm2_other += 1

    print("\nEmbedding length check")
    print("----------------------")
    print(f"Exact match:       {exact_match}")
    print(f"ESM2 +1 position:  {esm2_one_extra}")
    print(f"Other mismatch:    {esm2_other}")

    if examples:
        print("\nExamples:")
        for seq, seq_len, p_len, e_len in examples:
            print(
                f"{seq}: "
                f"sequence={seq_len}, "
                f"ProtT5={p_len}, "
                f"ESM2={e_len}"
            )
# ============================================================
# Dataset
# ============================================================

class FusionDataset(Dataset):

    def __init__(
        self,
        prott5_embeddings,
        esm2_embeddings
    ):
        self.prott5 = prott5_embeddings
        self.esm2 = esm2_embeddings

        # Only keep sequences available in BOTH models
        self.sequences = sorted(
            set(self.prott5.keys())
            & set(self.esm2.keys())
        )

        if len(self.sequences) == 0:
            raise ValueError(
                "No common sequences found between "
                "ProtT5 and ESM2 dictionaries."
            )

        print(
            f"Common sequences: {len(self.sequences)}"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        sequence = self.sequences[index]

        prott5 = np.asarray(
            self.prott5[sequence],
            dtype=np.float32
        )

        esm2 = np.asarray(
            self.esm2[sequence],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Sanity check
        # ----------------------------------------------------

        if prott5.ndim != 2:
            raise ValueError(
                f"ProtT5 embedding for {sequence} "
                f"must be 2D, got {prott5.shape}"
            )

        if esm2.ndim != 2:
            raise ValueError(
                f"ESM2 embedding for {sequence} "
                f"must be 2D, got {esm2.shape}"
            )

        if prott5.shape[0] != len(sequence):
            raise ValueError(
                f"ProtT5 length mismatch for {sequence}: "
                f"embedding length={prott5.shape[0]}, "
                f"sequence length={len(sequence)}"
            )

        if esm2.shape[0] != len(sequence):
            raise ValueError(
                f"ESM2 length mismatch for {sequence}: "
                f"embedding length={esm2.shape[0]}, "
                f"sequence length={len(sequence)}"
            )

        if prott5.shape[1] != PROTT5_DIM:
            raise ValueError(
                f"Expected ProtT5 dimension {PROTT5_DIM}, "
                f"got {prott5.shape[1]}"
            )

        if esm2.shape[1] != ESM2_DIM:
            raise ValueError(
                f"Expected ESM2 dimension {ESM2_DIM}, "
                f"got {esm2.shape[1]}"
            )

        return {
            "sequence": sequence,
            "prott5": torch.from_numpy(prott5),
            "esm2": torch.from_numpy(esm2)
        }


# ============================================================
# Padding collate function
# ============================================================

def collate_fn(batch):

    batch_size = len(batch)

    lengths = [
        item["prott5"].shape[0]
        for item in batch
    ]

    max_length = max(lengths)

    prott5 = torch.zeros(
        batch_size,
        max_length,
        PROTT5_DIM,
        dtype=torch.float32
    )

    esm2 = torch.zeros(
        batch_size,
        max_length,
        ESM2_DIM,
        dtype=torch.float32
    )

    padding_mask = torch.ones(
        batch_size,
        max_length,
        dtype=torch.bool
    )

    sequences = []

    for i, item in enumerate(batch):

        length = item["prott5"].shape[0]

        prott5[i, :length] = item["prott5"]
        esm2[i, :length] = item["esm2"]

        # False = valid position
        padding_mask[i, :length] = False

        sequences.append(item["sequence"])

    return {
        "sequence": sequences,
        "prott5": prott5,
        "esm2": esm2,
        "padding_mask": padding_mask
    }


# ============================================================
# Positional encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=512
    ):
        super().__init__()

        position = torch.arange(
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            )
            * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        length = x.size(1)

        return x + self.pe[:, :length]


# ============================================================
# Fusion Encoder
# ============================================================

class ProtT5ESM2FusionEncoder(nn.Module):

    def __init__(
        self,
        prott5_dim=1024,
        esm2_dim=1280,
        fusion_dim=1024,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        # ----------------------------------------------------
        # Project both modalities into common dimension
        # ----------------------------------------------------

        self.prott5_projection = nn.Linear(
            prott5_dim,
            fusion_dim
        )

        self.esm2_projection = nn.Linear(
            esm2_dim,
            fusion_dim
        )

        self.prott5_norm = nn.LayerNorm(
            fusion_dim
        )

        self.esm2_norm = nn.LayerNorm(
            fusion_dim
        )

        # ----------------------------------------------------
        # ProtT5 -> Query
        # ESM2   -> Key / Value
        # ----------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # ----------------------------------------------------
        # Feed-forward block
        # ----------------------------------------------------

        self.norm1 = nn.LayerNorm(
            fusion_dim
        )

        self.norm2 = nn.LayerNorm(
            fusion_dim
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                fusion_dim,
                fusion_dim * 2
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                fusion_dim * 2,
                fusion_dim
            ),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

        self.position = PositionalEncoding(
            fusion_dim
        )

    def forward(
        self,
        prott5,
        esm2,
        key_padding_mask=None
    ):

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------

        t = self.prott5_projection(
            prott5
        )

        e = self.esm2_projection(
            esm2
        )

        t = self.prott5_norm(t)
        e = self.esm2_norm(e)

        # ----------------------------------------------------
        # Positional information
        # ----------------------------------------------------

        t = self.position(t)
        e = self.position(e)

        # ----------------------------------------------------
        # Cross attention
        #
        # Q = ProtT5
        # K = ESM2
        # V = ESM2
        # ----------------------------------------------------

        attended, _ = self.cross_attention(
            query=t,
            key=e,
            value=e,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        h = self.norm1(
            t + self.dropout(attended)
        )

        # ----------------------------------------------------
        # Feed-forward
        # ----------------------------------------------------

        h = self.norm2(
            h + self.ffn(h)
        )

        return h


# ============================================================
# Decoder
# ============================================================

class SequenceDecoder(nn.Module):

    def __init__(
        self,
        fusion_dim=1024,
        num_heads=8,
        num_layers=2,
        ff_dim=2048,
        dropout=0.1,
        vocab_size=20
    ):
        super().__init__()

        self.vocab_size = vocab_size

        # ----------------------------------------------------
        # Amino-acid token embeddings
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            fusion_dim
        )

        self.position = PositionalEncoding(
            fusion_dim
        )

        # ----------------------------------------------------
        # Transformer decoder
        # ----------------------------------------------------

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # Predict amino acid
        # ----------------------------------------------------

        self.output = nn.Linear(
            fusion_dim,
            vocab_size
        )

    def forward(
        self,
        decoder_tokens,
        memory,
        memory_key_padding_mask=None
    ):

        x = self.token_embedding(
            decoder_tokens
        )

        x = self.position(x)

        x = self.decoder(
            tgt=x,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask
        )

        logits = self.output(x)

        return logits


# ============================================================
# Complete Fusion Autoencoder
# ============================================================

class FusionAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = ProtT5ESM2FusionEncoder(
            prott5_dim=PROTT5_DIM,
            esm2_dim=ESM2_DIM,
            fusion_dim=FUSION_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT
        )

        self.decoder = SequenceDecoder(
            fusion_dim=FUSION_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            ff_dim=FF_DIM,
            dropout=DROPOUT
        )

    def encode(
        self,
        prott5,
        esm2,
        padding_mask=None
    ):

        return self.encoder(
            prott5,
            esm2,
            key_padding_mask=padding_mask
        )

    def forward(
        self,
        prott5,
        esm2,
        decoder_tokens,
        padding_mask=None
    ):

        fused = self.encode(
            prott5,
            esm2,
            padding_mask
        )

        logits = self.decoder(
            decoder_tokens,
            fused,
            memory_key_padding_mask=padding_mask
        )

        return logits, fused


# ============================================================
# Amino-acid vocabulary
# ============================================================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

AA_TO_ID = {
    aa: i
    for i, aa in enumerate(AMINO_ACIDS)
}

ID_TO_AA = {
    i: aa
    for i, aa in enumerate(AMINO_ACIDS)
}


# ============================================================
# Masking
# ============================================================

def create_masked_inputs(
    sequences,
    prott5,
    esm2,
    padding_mask,
    mask_probability=0.15
):
    """
    Create masked inputs.

    At masked positions, BOTH ProtT5 and ESM2 embeddings
    are replaced by zero.

    The decoder target remains the original amino acid.

    This prevents the model from simply reading the original
    residue embedding at the position it is supposed to predict.
    """

    batch_size, max_length, _ = prott5.shape

    masked_prott5 = prott5.clone()
    masked_esm2 = esm2.clone()

    target = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long,
        device=prott5.device
    )

    mask_positions = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.bool,
        device=prott5.device
    )

    for i, sequence in enumerate(sequences):

        length = len(sequence)

        for j in range(length):

            if random.random() < mask_probability:

                aa = sequence[j]

                target[i, j] = AA_TO_ID[aa]

                mask_positions[i, j] = True

    # --------------------------------------------------------
    # Guarantee at least one masked residue per sequence
    # --------------------------------------------------------

    for i, sequence in enumerate(sequences):

        if not mask_positions[i, :len(sequence)].any():

            j = random.randrange(
                len(sequence)
            )

            aa = sequence[j]

            target[i, j] = AA_TO_ID[aa]

            mask_positions[i, j] = True

    # --------------------------------------------------------
    # Hide embeddings at masked positions
    # --------------------------------------------------------

    masked_prott5[
        mask_positions
    ] = 0.0

    masked_esm2[
        mask_positions
    ] = 0.0

    return (
        masked_prott5,
        masked_esm2,
        target,
        mask_positions
    )


# ============================================================
# Decoder input
# ============================================================

def create_decoder_input(
    sequences,
    mask_positions,
    max_length,
    device
):
    """
    Decoder receives a MASK token representation at masked
    positions and a neutral token at other positions.

    Since the reconstruction loss is calculated only at masked
    positions, the important signal is the encoder memory.
    """

    # 0 is used as a generic decoder token.
    decoder_tokens = torch.zeros(
        len(sequences),
        max_length,
        dtype=torch.long,
        device=device
    )

    return decoder_tokens


# ============================================================
# Loss
# ============================================================

def masked_reconstruction_loss(
    logits,
    targets
):
    """
    Cross entropy only on masked amino-acid positions.
    """

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100
    )


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch in loader:

        prott5 = batch["prott5"].to(
            DEVICE
        )

        esm2 = batch["esm2"].to(
            DEVICE
        )

        padding_mask = batch["padding_mask"].to(
            DEVICE
        )

        sequences = batch["sequence"]

        # ----------------------------------------------------
        # Create masked embeddings
        # ----------------------------------------------------

        (
            masked_prott5,
            masked_esm2,
            targets,
            mask_positions
        ) = create_masked_inputs(
            sequences,
            prott5,
            esm2,
            padding_mask,
            MASK_PROB
        )

        # ----------------------------------------------------
        # Decoder input
        # ----------------------------------------------------

        decoder_tokens = create_decoder_input(
            sequences,
            mask_positions,
            prott5.size(1),
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits, _ = model(
            masked_prott5,
            masked_esm2,
            decoder_tokens,
            padding_mask
        )

        # ----------------------------------------------------
        # Reconstruction loss
        # ----------------------------------------------------

        loss = masked_reconstruction_loss(
            logits,
            targets
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader
):

    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in loader:

        prott5 = batch["prott5"].to(
            DEVICE
        )

        esm2 = batch["esm2"].to(
            DEVICE
        )

        padding_mask = batch["padding_mask"].to(
            DEVICE
        )

        sequences = batch["sequence"]

        (
            masked_prott5,
            masked_esm2,
            targets,
            mask_positions
        ) = create_masked_inputs(
            sequences,
            prott5,
            esm2,
            padding_mask,
            MASK_PROB
        )

        decoder_tokens = create_decoder_input(
            sequences,
            mask_positions,
            prott5.size(1),
            DEVICE
        )

        logits, _ = model(
            masked_prott5,
            masked_esm2,
            decoder_tokens,
            padding_mask
        )

        loss = masked_reconstruction_loss(
            logits,
            targets
        )

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


# ============================================================
# Extract fused embeddings
# ============================================================
@torch.no_grad()
def extract_fused_embeddings(
    model,
    dataset
):
    """
    Extract UNMASKED fused embeddings.

    IMPORTANT:
        No pooling.

    Returns
    -------
    sequences : list
        Protein/peptide sequences.

    embeddings : list
        Corresponding residue-level embeddings
        with shape [L, 1024].
    """

    model.eval()

    sequences = []
    embeddings = []

    for index in range(len(dataset)):

        item = dataset[index]

        sequence = item["sequence"]

        prott5 = item["prott5"].unsqueeze(0).to(
            DEVICE
        )

        esm2 = item["esm2"].unsqueeze(0).to(
            DEVICE
        )

        padding_mask = torch.zeros(
            1,
            len(sequence),
            dtype=torch.bool,
            device=DEVICE
        )

        # ----------------------------------------------------
        # No masking during final extraction.
        # No pooling.
        # ----------------------------------------------------

        fused = model.encode(
            prott5,
            esm2,
            padding_mask
        )

        # [1, L, 1024] -> [L, 1024]

        fused = fused.squeeze(0).cpu().numpy()

        sequences.append(sequence)
        embeddings.append(fused)

    return sequences, embeddings

def train_fusion_model(
    prott5_embeddings,
    esm2_embeddings
):
    """
    Train the self-supervised fusion model.

    Returns
    -------
    model : FusionAutoencoder
        Best trained fusion model.
    dataset : FusionDataset
        Complete dataset used for training/inference.
    """

    # --------------------------------------------------------
    # Check embedding consistency
    # --------------------------------------------------------

    check_embedding_lengths(
        prott5_embeddings,
        esm2_embeddings
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = FusionDataset(
        prott5_embeddings,
        esm2_embeddings
    )

    # --------------------------------------------------------
    # Train / validation split
    #
    # Used only for self-supervised reconstruction.
    # No AMP/MIC/activity values are used.
    # --------------------------------------------------------

    indices = list(range(len(dataset)))

    random.shuffle(indices)

    validation_size = max(
        1,
        int(0.1 * len(indices))
    )

    val_indices = indices[
        :validation_size
    ]

    train_indices = indices[
        validation_size:
    ]

    train_dataset = torch.utils.data.Subset(
        dataset,
        train_indices
    )

    val_dataset = torch.utils.data.Subset(
        dataset,
        val_indices
    )

    print(
        f"\nTraining sequences: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation sequences: "
        f"{len(val_dataset)}"
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FusionAutoencoder().to(
        DEVICE
    )

    print(
        f"\nModel parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=MIN_DELTA
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\nStarting self-supervised training...\n"
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = validate(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Check validation improvement
        # ----------------------------------------------------

        if val_loss < (
            early_stopping.best_loss - MIN_DELTA
        ):

            early_stopping.best_loss = val_loss
            early_stopping.counter = 0

            # ------------------------------------------------
            # Save best model
            # ------------------------------------------------

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": {
                        "prott5_dim": PROTT5_DIM,
                        "esm2_dim": ESM2_DIM,
                        "fusion_dim": FUSION_DIM,
                        "num_heads": NUM_HEADS,
                        "num_layers": NUM_LAYERS,
                        "ff_dim": FF_DIM,
                    }
                },
                BEST_MODEL_FILE
            )

            print(
                f"  -> Saved best model: "
                f"{BEST_MODEL_FILE}"
            )

        else:

            early_stopping.counter += 1

            print(
                f"  -> Early stopping counter: "
                f"{early_stopping.counter}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            early_stopping.counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"Best validation loss: "
                f"{early_stopping.best_loss:.6f}"
            )

            break

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not BEST_MODEL_FILE.exists():

        raise RuntimeError(
            "No best model checkpoint was saved."
        )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print(
        "\nLoading best fusion model..."
    )

    checkpoint = torch.load(
        BEST_MODEL_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{checkpoint['val_loss']:.6f}"
    )

    return model, dataset

def infer_fused_embeddings(
    checkpoint_path,
    prott5_file,
    esm2_file,
    output_file
):
    """
    Load a trained fusion model and extract
    residue-level fused embeddings.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the trained fusion model checkpoint.

    prott5_file : str or Path
        ProtT5 embedding file.

    esm2_file : str or Path
        ESM2 embedding file.

    output_file : str or Path
        Output fused embedding file.

    Returns
    -------
    sequences : list
        Protein/peptide sequences.

    embeddings : list
        Fused embeddings with shape [L, 1024].
    """

    # --------------------------------------------------------
    # Load ProtT5 embeddings
    # --------------------------------------------------------

    print("\nLoading ProtT5 embeddings...")

    prott5_embeddings = load_embeddings(
        prott5_file
    )

    print(
        f"ProtT5 sequences: "
        f"{len(prott5_embeddings)}"
    )

    # --------------------------------------------------------
    # Load ESM2 embeddings
    # --------------------------------------------------------

    print("\nLoading ESM2 embeddings...")

    esm2_embeddings = load_embeddings(
        esm2_file
    )

    print(
        f"ESM2 sequences: "
        f"{len(esm2_embeddings)}"
    )

    # --------------------------------------------------------
    # Check embedding consistency
    # --------------------------------------------------------

    check_embedding_lengths(
        prott5_embeddings,
        esm2_embeddings
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = FusionDataset(
        prott5_embeddings,
        esm2_embeddings
    )

    print(
        f"\nInference sequences: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = FusionAutoencoder().to(
        DEVICE
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Fusion model checkpoint not found: "
            f"{checkpoint_path}"
        )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print(
        f"\nLoading fusion model from:"
        f"\n{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded model from epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint['val_loss']:.6f}"
    )

    # --------------------------------------------------------
    # Extract fused embeddings
    # --------------------------------------------------------

    print(
        "\nExtracting fused embeddings..."
    )

    sequences, embeddings = extract_fused_embeddings(
        model,
        dataset
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    data_util.save_embeddings(
        sequences,
        embeddings,
        output_file
    )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    first_sequence = sequences[0]
    first_embedding = embeddings[0]

    print("\nExample:")

    print(
        f"Sequence: "
        f"{first_sequence}"
    )

    print(
        f"Shape: "
        f"{first_embedding.shape}"
    )

    print(
        "\nExpected shape:"
        "\n(L, 1024)"
    )

    return sequences, embeddings

# ============================================================
# Main
# ============================================================
def main():

    # --------------------------------------------------------
    # Load ProtT5 embeddings
    # --------------------------------------------------------

    '''print("\nLoading ProtT5 embeddings...")

    prott5_embeddings = load_embeddings(
        PROTT5_FILE
    )

    print(
        f"ProtT5 sequences: "
        f"{len(prott5_embeddings)}"
    )

    # --------------------------------------------------------
    # Load ESM2 embeddings
    # --------------------------------------------------------

    print("\nLoading ESM2 embeddings...")

    esm2_embeddings = load_embeddings(
        ESM2_FILE
    )

    print(
        f"ESM2 sequences: "
        f"{len(esm2_embeddings)}"
    )

    # --------------------------------------------------------
    # Train fusion model
    # --------------------------------------------------------

    model, dataset = train_fusion_model(
        prott5_embeddings,
        esm2_embeddings
    )'''

    # --------------------------------------------------------
    # Inference + save embeddings
    # --------------------------------------------------------

    infer_fused_embeddings(
        checkpoint_path=(
            "./fusion_result/fusion_model_ESM2_ProtT5/best_fusion_model.pt"
        ),
        prott5_file=PROTT5_FILE,
        esm2_file=ESM2_FILE,
        output_file=FUSED_EMBEDDINGS_FILE
    )
    
# ============================================================
# Early stopping
# ============================================================
class EarlyStopping:

    def __init__(
        self,
        patience=15,
        min_delta=1e-4
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0

if __name__ == "__main__":
    main()