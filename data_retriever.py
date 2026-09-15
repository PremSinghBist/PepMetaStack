import pickle
import numpy as np
import pandas as pd

from blosum62 import load_blosum62_features
import physiochem_feature_extractor as PFE
import position_encoding_extractor as PEE


def load_features(
    fused_embeddings=None,
    blosum_csv="",
    physio_csv="",
    sinusoidal_csv="",
    dataset_csv="",
    fold_index=1,
):
    """
    Load GMProt/fused embeddings and additional features using
    the sequence as the common key.

    Create train/validation/test splits for the selected CV fold.

    split == "test"       -> fixed test set
    cv_fold == fold_index -> validation
    other cv_fold values  -> training
    """

    # ==========================================================
    # Validate fold
    # ==========================================================

    if fold_index not in [1, 2, 3, 4, 5]:
        raise ValueError("fold_index must be between 1 and 5.")

    # ==========================================================
    # Load dataset metadata
    # ==========================================================

    df = pd.read_csv(dataset_csv)

    required = {
        "sequence",
        "value",
        "split",
        "cv_fold",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # ==========================================================
    # Convert metadata to dictionary using sequence as key
    # ==========================================================

    metadata = (
        df.set_index("sequence")[["value", "split", "cv_fold"]]
        .to_dict("index")
    )



    if not isinstance(fused_embeddings, dict):
        raise ValueError(
            "fused_embedding_file must contain a dictionary "
            "of {sequence: embedding_array}."
        )

    # ==========================================================
    # Load additional features
    # ==========================================================

    blosum_dict = load_blosum62_features(
        csv_path=blosum_csv
    )

    physio_dict = PFE.load_physio_features_as_numpy_all(
        physio_csv
    )

    sinusoidal_dict = PEE.load_sinusoidal_encoding(
        sinusoidal_csv
    )

    # ==========================================================
    # Create splits
    # ==========================================================

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    # ==========================================================
    # Use dataset sequences as the master sequence list
    # ==========================================================

    for seq in df["sequence"]:

        # ------------------------------------------------------
        # Check GMProt embedding
        # ------------------------------------------------------

        if seq not in fused_embeddings:
            raise ValueError(
                f"{seq} missing in GMProt/fused embeddings."
            )

        # ------------------------------------------------------
        # Check additional features
        # ------------------------------------------------------

        if seq not in physio_dict:
            raise ValueError(
                f"{seq} missing in physicochemical features."
            )

        if seq not in blosum_dict:
            raise ValueError(
                f"{seq} missing in BLOSUM features."
            )

        if seq not in sinusoidal_dict:
            raise ValueError(
                f"{seq} missing in sinusoidal features."
            )

        row = metadata[seq]

        # ======================================================
        # Determine split
        # ======================================================

        if row["split"] == "test":

            split = "test"
            cv_fold = None

        elif int(row["cv_fold"]) == fold_index:

            split = "val"
            cv_fold = int(row["cv_fold"])

        else:

            split = "train"
            cv_fold = int(row["cv_fold"])

        # ======================================================
        # GMProt / Fused Embedding
        # ======================================================

        emb = np.asarray(
            fused_embeddings[seq],
            dtype=np.float32,
        ).reshape(-1)

        # ======================================================
        # Additional Features
        # ======================================================

        physio = np.asarray(
            physio_dict[seq],
            dtype=np.float32,
        )

        blosum = np.asarray(
            blosum_dict[seq],
            dtype=np.float32,
        )

        sino = np.asarray(
            sinusoidal_dict[seq],
            dtype=np.float32,
        )

        # ======================================================
        # Store Record
        # ======================================================

        splits[split].append(
            {
                "emb": emb,
                "physio": physio,
                "blosum": blosum,
                "sino": sino,
                "seq": seq,
                "mic": row["value"],
                "split": split,
                "cv_fold": cv_fold,
            }
        )

    # ==========================================================
    # Print Split Sizes
    # ==========================================================

    print(
        f"\nCV Fold {fold_index}: "
        f"Train={len(splits['train'])}, "
        f"Val={len(splits['val'])}, "
        f"Test={len(splits['test'])}"
    )

    return splits
    

if __name__ == "__main__":
    dataset_path = "ecoli_prott5_based_datasets.pkl"
    data_retriever = DataRetriever(dataset_path)
    data_retriever.explain_data_structure(total_samples=10)  # Inspect first 3 samples of Fold 1 Train set
    