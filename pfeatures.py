#!/usr/bin/env python3

import sys
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

HEMOPI2_DIR = (
    BASE_DIR / "hemolyis_model"
).resolve()

PFEATURE_SCRIPT = (
    HEMOPI2_DIR /
    "composition_calculate_hemopi2_2.py"
).resolve()

PFEATURE_DATA_DIR = (
    HEMOPI2_DIR /
    "Data"
).resolve()

SEQUENCE_COLUMN = "sequence"

# Maximum sequence length used in this study
MAX_SEQUENCE_LENGTH = 128

VALID_AMINO_ACIDS = set(
    "ACDEFGHIKLMNPQRSTVWY"
)


# ============================================================
# 1. ENVIRONMENT VALIDATION
# ============================================================

def _validate_environment():
    """
    Validate HemoPI2 installation and required reference files.
    """

    if not HEMOPI2_DIR.exists():
        raise FileNotFoundError(
            f"HemoPI2 directory not found:\n"
            f"{HEMOPI2_DIR}"
        )

    if not PFEATURE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Pfeature extractor not found:\n"
            f"{PFEATURE_SCRIPT}"
        )

    if not PFEATURE_DATA_DIR.exists():
        raise FileNotFoundError(
            f"HemoPI2 Data directory not found:\n"
            f"{PFEATURE_DATA_DIR}"
        )

    required_files = [
        "PhysicoChemical.csv",
        "atom.csv",
        "bonds.csv",
        "aa_attr_group.csv",
    ]

    missing_files = []

    for filename in required_files:

        path = (
            PFEATURE_DATA_DIR /
            filename
        )

        if not path.exists():
            missing_files.append(path)

    if missing_files:

        message = (
            "Required HemoPI2 reference files are missing:\n"
        )

        for path in missing_files:
            message += f"  - {path}\n"

        raise FileNotFoundError(
            message
        )


# ============================================================
# 2. HEMOPI2 PATH COMPATIBILITY
# ============================================================

def _prepare_data_path():
    """
    The original HemoPI2 code expects reference files under:

        Model/Data/

    Create a symbolic link:

        hemolyis_model/Model/Data
                    ->
        hemolyis_model/Data
    """

    model_dir = (
        HEMOPI2_DIR /
        "Model"
    )

    expected_data_dir = (
        model_dir /
        "Data"
    )

    if expected_data_dir.exists():
        return

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        expected_data_dir.symlink_to(
            PFEATURE_DATA_DIR,
            target_is_directory=True,
        )

    except OSError as exc:

        raise RuntimeError(
            "Unable to create HemoPI2 "
            "Model/Data compatibility path.\n\n"
            f"Expected:\n{expected_data_dir}\n\n"
            f"Target:\n{PFEATURE_DATA_DIR}"
        ) from exc


# ============================================================
# 3. LOAD AND VALIDATE SEQUENCES
# ============================================================

def _load_sequences(csv_path):
    """
    Load and validate peptide sequences.

    Notes
    -----
    - Sequences containing only standard amino acids are retained.
    - Sequences of length 1 or 2 are valid and are NOT removed.
    - Length-dependent HemoPI2 descriptors that cannot be
      calculated for these sequences must return 0.0.
    - Sequences longer than MAX_SEQUENCE_LENGTH are truncated.

    Returns
    -------
    original_sequences : list[str]
        Original cleaned sequences used as dictionary keys.

    processed_sequences : list[str]
        Sequences supplied to HemoPI2 after maximum-length
        preprocessing.
    """

    csv_path = Path(
        csv_path
    ).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n"
            f"{csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    # --------------------------------------------------------
    # Check sequence column
    # --------------------------------------------------------

    if SEQUENCE_COLUMN not in df.columns:

        raise ValueError(
            f"Column '{SEQUENCE_COLUMN}' "
            f"was not found in:\n"
            f"{csv_path}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_mask = (
        df[SEQUENCE_COLUMN].isna()
    )

    if missing_mask.any():

        rows = (
            df.index[
                missing_mask
            ]
            .tolist()
        )

        raise ValueError(
            "Missing sequences detected at "
            f"rows:\n{rows}"
        )

    # --------------------------------------------------------
    # Clean sequences
    # --------------------------------------------------------

    sequences = (
        df[SEQUENCE_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Empty sequences
    # --------------------------------------------------------

    empty_mask = (
        sequences.str.len() == 0
    )

    if empty_mask.any():

        rows = (
            df.index[
                empty_mask
            ]
            .tolist()
        )

        raise ValueError(
            "Empty peptide sequences "
            f"detected at rows:\n{rows}"
        )

    # --------------------------------------------------------
    # Validate amino-acid alphabet
    # --------------------------------------------------------

    invalid = []

    for index, sequence in sequences.items():

        invalid_residues = (
            set(sequence)
            -
            VALID_AMINO_ACIDS
        )

        if invalid_residues:

            invalid.append(
                (
                    index,
                    sequence,
                    sorted(
                        invalid_residues
                    ),
                )
            )

    if invalid:

        message = (
            "Invalid amino-acid residues detected.\n\n"
        )

        for (
            index,
            sequence,
            residues,
        ) in invalid[:20]:

            message += (
                f"row={index}, "
                f"sequence={sequence}, "
                f"invalid={residues}\n"
            )

        raise ValueError(
            message
        )

    # --------------------------------------------------------
    # Short sequences
    #
    # Keep them.
    # Only undefined length-dependent descriptors are assigned
    # zero by the modified HemoPI2 extractor.
    # --------------------------------------------------------

    short_mask = (
        sequences.str.len() < 3
    )

    short_count = int(
        short_mask.sum()
    )

    if short_count > 0:

        short_rows = (
            df.index[
                short_mask
            ]
            .tolist()
        )

        print(
            f"\nWARNING: {short_count} sequences "
            "have length < 3 aa."
        )

        print(
            "These sequences are retained."
        )

        print(
            "Undefined pair/triad/lag descriptors "
            "will be represented by 0.0."
        )

        print(
            f"Rows: {short_rows}"
        )

    # --------------------------------------------------------
    # Remove duplicate sequences
    #
    # Output is sequence -> vector, so identical sequences
    # require only one dictionary entry.
    # --------------------------------------------------------

    duplicated_count = int(
        sequences.duplicated().sum()
    )

    if duplicated_count:

        print(
            f"\nDuplicate sequences removed: "
            f"{duplicated_count}"
        )

        sequences = (
            sequences
            .drop_duplicates(
                keep="first"
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # Preserve original sequence for dictionary key
    # --------------------------------------------------------

    original_sequences = (
        sequences.tolist()
    )

    # --------------------------------------------------------
    # Maximum-length preprocessing
    # --------------------------------------------------------

    processed_sequences = [
        sequence[
            :MAX_SEQUENCE_LENGTH
        ]
        for sequence
        in original_sequences
    ]

    truncated_count = sum(
        len(sequence)
        >
        MAX_SEQUENCE_LENGTH

        for sequence
        in original_sequences
    )

    lengths = [
        len(sequence)
        for sequence
        in original_sequences
    ]

    processed_lengths = [
        len(sequence)
        for sequence
        in processed_sequences
    ]

    print(
        "\nDataset summary"
    )

    print(
        "-" * 50
    )

    print(
        f"Sequences              : "
        f"{len(original_sequences)}"
    )

    print(
        f"Original length range  : "
        f"{min(lengths)}-{max(lengths)} aa"
    )

    print(
        f"Processed length range : "
        f"{min(processed_lengths)}-"
        f"{max(processed_lengths)} aa"
    )

    print(
        f"Maximum allowed length : "
        f"{MAX_SEQUENCE_LENGTH} aa"
    )

    print(
        f"Sequences truncated    : "
        f"{truncated_count}"
    )

    print(
        f"Sequences <3 aa        : "
        f"{short_count}"
    )

    return (
        original_sequences,
        processed_sequences,
    )


# ============================================================
# 4. RUN ORIGINAL HEMOPI2 PFEATURE EXTRACTOR
# ============================================================

def _run_pfeature_extractor(
    processed_sequences,
):
    """
    Run the HemoPI2 Pfeature extractor.
    """

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="pfeature_",
        )
    ).resolve()

    sequence_file = (
        temp_dir /
        "sequences.txt"
    )

    pfeature_file = (
        temp_dir /
        "pfeatures.csv"
    )

    try:

        # --------------------------------------------------------
        # HemoPI2 input:
        # one sequence per line
        # --------------------------------------------------------

        pd.Series(
            processed_sequences
        ).to_csv(
            sequence_file,
            index=False,
            header=False,
        )

        command = [
            sys.executable,
            str(PFEATURE_SCRIPT),
            str(sequence_file),
            str(temp_dir),
            str(pfeature_file),
        ]

        print(
            "\nComputing Pfeatures..."
        )

        subprocess.run(
            command,
            cwd=str(
                HEMOPI2_DIR
            ),
            check=True,
        )

        if not pfeature_file.exists():

            raise RuntimeError(
                "HemoPI2 completed but did not "
                "produce the expected output:\n"
                f"{pfeature_file}"
            )

        # --------------------------------------------------------
        # Load generated features
        # --------------------------------------------------------

        pfeature_df = pd.read_csv(
            pfeature_file
        )

        # --------------------------------------------------------
        # Remove accidental unnamed/index columns
        # --------------------------------------------------------

        pfeature_df = pfeature_df.loc[
            :,
            ~pfeature_df.columns
            .astype(str)
            .str.startswith(
                "Unnamed:"
            )
        ]

        return (
            pfeature_df.copy()
        )

    except subprocess.CalledProcessError as exc:

        raise RuntimeError(
            "Pfeature extraction failed.\n\n"
            f"Command:\n"
            f"{' '.join(command)}\n\n"
            f"Exit code: "
            f"{exc.returncode}"
        ) from exc

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


# ============================================================
# 5. VALIDATE FEATURE MATRIX
# ============================================================

def _validate_features(
    pfeature_df,
    sequences,
):
    """
    Convert HemoPI2 output into a validated float32 matrix.
    """

    # --------------------------------------------------------
    # Row alignment
    # --------------------------------------------------------

    if (
        len(pfeature_df)
        !=
        len(sequences)
    ):

        raise RuntimeError(
            "Sequence/Pfeature row mismatch.\n\n"
            f"Sequences : "
            f"{len(sequences)}\n"
            f"Pfeatures : "
            f"{len(pfeature_df)}"
        )

    # --------------------------------------------------------
    # Convert all columns to numeric
    # --------------------------------------------------------

    numeric_df = (
        pfeature_df.apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    feature_matrix = np.asarray(
        numeric_df.to_numpy(),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Matrix validation
    # --------------------------------------------------------

    if feature_matrix.ndim != 2:

        raise RuntimeError(
            "Expected a 2-dimensional "
            "Pfeature matrix, but received "
            f"{feature_matrix.shape}."
        )

    if feature_matrix.shape[1] == 0:

        raise RuntimeError(
            "Pfeature extractor returned "
            "zero features."
        )

    # --------------------------------------------------------
    # NaN check
    # --------------------------------------------------------

    nan_mask = np.isnan(
        feature_matrix
    )

    nan_count = int(
        nan_mask.sum()
    )

    if nan_count > 0:

        rows, cols = np.where(
            nan_mask
        )

        examples = []

        for row, col in zip(
            rows[:10],
            cols[:10],
        ):

            examples.append(
                (
                    sequences[
                        int(row)
                    ],
                    str(
                        numeric_df.columns[
                            int(col)
                        ]
                    ),
                )
            )

        message = (
            f"Pfeature matrix contains "
            f"{nan_count} NaN values.\n\n"
            "First affected sequence/features:\n"
        )

        for sequence, feature in examples:

            message += (
                f"  sequence={sequence} | "
                f"feature={feature}\n"
            )

        message += (
            "\nThe HemoPI2 short-sequence "
            "guards may be incomplete."
        )

        raise ValueError(
            message
        )

    # --------------------------------------------------------
    # Infinite-value check
    # --------------------------------------------------------

    inf_count = int(
        np.isinf(
            feature_matrix
        ).sum()
    )

    if inf_count > 0:

        raise ValueError(
            f"Pfeature matrix contains "
            f"{inf_count} infinite values."
        )

    print(
        "\nPfeature matrix validated"
    )

    print(
        "-" * 50
    )

    print(
        f"Samples       : "
        f"{feature_matrix.shape[0]}"
    )

    print(
        f"Features      : "
        f"{feature_matrix.shape[1]}"
    )

    print(
        f"dtype         : "
        f"{feature_matrix.dtype}"
    )

    print(
        f"NaN values    : "
        f"{nan_count}"
    )

    print(
        f"Inf values    : "
        f"{inf_count}"
    )

    return feature_matrix


# ============================================================
# 6. BUILD SEQUENCE -> FEATURE DICTIONARY
# ============================================================

def _build_feature_dictionary(
    sequences,
    feature_matrix,
):
    """
    Construct:

        sequence -> np.ndarray
    """

    if (
        len(sequences)
        !=
        feature_matrix.shape[0]
    ):

        raise RuntimeError(
            "Cannot create feature dictionary: "
            "sequence and feature counts differ."
        )

    feature_dict = {}

    for (
        sequence,
        vector,
    ) in zip(
        sequences,
        feature_matrix,
    ):

        feature_dict[
            sequence
        ] = np.asarray(
            vector,
            dtype=np.float32,
        )

    return feature_dict


# ============================================================
# PUBLIC METHOD
# ============================================================

def compute_and_save_pfeatures(
    csv_path,
    output_path,
):
    """
    Compute HemoPI2/Pfeature descriptors and save:

        sequence -> feature vector

    Parameters
    ----------
    csv_path : str or Path
        CSV containing a column named "sequence".

    output_path : str or Path
        Destination .npy file.

    Returns
    -------
    dict
        Dictionary mapping each peptide sequence to its
        float32 Pfeature vector.
    """

    print(
        "\n" +
        "=" * 70
    )

    print(
        "PFEATURE EXTRACTION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Resolve output path
    # --------------------------------------------------------

    output_path = Path(
        output_path
    ).resolve()

    if (
        output_path.suffix.lower()
        !=
        ".npy"
    ):

        output_path = (
            output_path.with_suffix(
                ".npy"
            )
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    _validate_environment()

    _prepare_data_path()

    # --------------------------------------------------------
    # Load sequences
    # --------------------------------------------------------

    (
        original_sequences,
        processed_sequences,
    ) = _load_sequences(
        csv_path
    )

    # --------------------------------------------------------
    # Compute descriptors
    # --------------------------------------------------------

    pfeature_df = (
        _run_pfeature_extractor(
            processed_sequences
        )
    )

    # --------------------------------------------------------
    # Validate descriptors
    # --------------------------------------------------------

    feature_matrix = (
        _validate_features(
            pfeature_df=
                pfeature_df,

            sequences=
                original_sequences,
        )
    )

    # --------------------------------------------------------
    # Build sequence -> vector mapping
    # --------------------------------------------------------

    feature_dict = (
        _build_feature_dictionary(
            sequences=
                original_sequences,

            feature_matrix=
                feature_matrix,
        )
    )

    # --------------------------------------------------------
    # Save dictionary
    # --------------------------------------------------------

    np.save(
        output_path,
        feature_dict,
        allow_pickle=True,
    )

    # --------------------------------------------------------
    # Reload and verify
    # --------------------------------------------------------

    loaded_dict = (
        np.load(
            output_path,
            allow_pickle=True,
        )
        .item()
    )

    if (
        len(loaded_dict)
        !=
        len(feature_dict)
    ):

        raise RuntimeError(
            "Saved Pfeature dictionary "
            "verification failed."
        )

    # --------------------------------------------------------
    # Verify feature dimensions
    # --------------------------------------------------------

    dimensions = {
        vector.shape
        for vector
        in loaded_dict.values()
    }

    if len(dimensions) != 1:

        raise RuntimeError(
            "Saved Pfeature vectors have "
            "inconsistent dimensions:\n"
            f"{dimensions}"
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n" +
        "=" * 70
    )

    print(
        "PFEATURE EXTRACTION COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Sequences    : "
        f"{len(feature_dict)}"
    )

    print(
        f"Feature size : "
        f"{feature_matrix.shape[1]}"
    )

    print(
        f"Saved file   : "
        f"{output_path}"
    )

    first_sequence = next(
        iter(
            feature_dict
        )
    )

    print(
        f"Example      : "
        f"{first_sequence}"
    )

    print(
        f"Vector shape : "
        f"{feature_dict[first_sequence].shape}"
    )

    return feature_dict


# ============================================================
# OPTIONAL LOADER
# ============================================================

def load_pfeatures(
    file_path,
):
    """
    Load saved Pfeature dictionary.
    """

    pfeatures = (
        np.load(
            file_path,
            allow_pickle=True,
        )
        .item()
    )

    if not pfeatures:
        raise ValueError(
            "Pfeature dictionary is empty."
        )

    first_sequence = next(
        iter(
            pfeatures
        )
    )

    feature_vector = (
        pfeatures[
            first_sequence
        ]
    )

    print(
        f"Loaded Pfeatures from:\n"
        f"{file_path}"
    )

    print(
        f"Sequences    : "
        f"{len(pfeatures)}"
    )

    print(
        f"Feature size : "
        f"{feature_vector.shape}"
    )

    return pfeatures


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    output_path = (
        "./data/ecoli_pfeatures.npy"
    )

    feature_dict = (
        compute_and_save_pfeatures(
            csv_path=(
                "./data/"
                "ecoli_five_fold_dataset.csv"
            ),
            output_path=
                output_path,
        )
    )

    load_pfeatures(
        output_path
    )