#!/usr/bin/env python3

import sys
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

# Directory containing this script
BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Original HemoPI2 feature extractor
# ------------------------------------------------------------

HEMOLYSIS_MODEL_DIR = (
    BASE_DIR / "hemolyis_model"
).resolve()

PFEATURE_SCRIPT = (
    HEMOLYSIS_MODEL_DIR /
    "composition_calculate_hemopi2_2.py"
).resolve()

# Actual location of HemoPI2 reference files
PFEATURE_DATA_DIR = (
    HEMOLYSIS_MODEL_DIR /
    "Data"
).resolve()


# ------------------------------------------------------------
# Input datasets
# ------------------------------------------------------------

DATASETS = {
    "crossvalidation": (
        BASE_DIR /
        "data" /
        "hemolysis" /
        "cross_val_dataset.csv"
    ).resolve(),

    "independent_test": (
        BASE_DIR /
        "data" /
        "hemolysis" /
        "independent_dataset.csv"
    ).resolve(),
}


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = (
    BASE_DIR /
    "data" /
    "hemolysis" /
    "pfeatures"
).resolve()


# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------

SEQUENCE_COLUMN = "SEQUENCE"

# HemoPI2 preprocessing truncates sequences longer than 40 aa
MAX_LENGTH = 40

# 20 standard amino acids
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ============================================================
# Validate HemoPI2 environment
# ============================================================

def validate_pfeature_environment():
    """
    Check that the HemoPI2 Pfeature extractor and its
    reference-data directory are available.
    """

    print("\nChecking HemoPI2 environment...")

    # --------------------------------------------------------
    # Check model directory
    # --------------------------------------------------------

    if not HEMOLYSIS_MODEL_DIR.exists():
        raise FileNotFoundError(
            "\nHemoPI2 model directory not found:\n"
            f"{HEMOLYSIS_MODEL_DIR}"
        )

    # --------------------------------------------------------
    # Check extractor
    # --------------------------------------------------------

    if not PFEATURE_SCRIPT.exists():
        raise FileNotFoundError(
            "\nPfeature extractor not found:\n"
            f"{PFEATURE_SCRIPT}"
        )

    # --------------------------------------------------------
    # Check Data directory
    # --------------------------------------------------------

    if not PFEATURE_DATA_DIR.exists():
        raise FileNotFoundError(
            "\nHemoPI2 Data directory not found:\n"
            f"{PFEATURE_DATA_DIR}"
        )

    # --------------------------------------------------------
    # Check important reference files
    # --------------------------------------------------------

    required_files = [
        "PhysicoChemical.csv",
        "atom.csv",
        "bonds.csv",
        "aa_attr_group.csv",
    ]

    missing_files = []

    for filename in required_files:

        file_path = PFEATURE_DATA_DIR / filename

        if not file_path.exists():
            missing_files.append(file_path)

    if missing_files:

        message = (
            "\nRequired HemoPI2 reference files are missing:\n"
        )

        for file_path in missing_files:
            message += f"  - {file_path}\n"

        raise FileNotFoundError(message)

    print("HemoPI2 environment OK")
    print(f"Model directory : {HEMOLYSIS_MODEL_DIR}")
    print(f"Extractor       : {PFEATURE_SCRIPT}")
    print(f"Data directory  : {PFEATURE_DATA_DIR}")


# ============================================================
# Compatibility for original HemoPI2 paths
# ============================================================

def prepare_hemopi2_path_compatibility():
    """
    The original composition_calculate_hemopi2_2.py uses
    relative paths such as:

        Model/Data/PhysicoChemical.csv

    However, in the current project the reference files are:

        hemolyis_model/Data/PhysicoChemical.csv

    We therefore create:

        hemolyis_model/Model/Data -> hemolyis_model/Data

    as a symbolic link.

    This allows the original HemoPI2 feature script to remain
    completely unchanged.
    """

    model_dir = HEMOLYSIS_MODEL_DIR / "Model"

    expected_data_path = model_dir / "Data"

    # --------------------------------------------------------
    # Already exists
    # --------------------------------------------------------

    if expected_data_path.exists():

        print(
            "\nHemoPI2 compatibility path already exists:"
        )
        print(expected_data_path)

        return

    # --------------------------------------------------------
    # Create Model directory
    # --------------------------------------------------------

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create:
    #
    # hemolyis_model/Model/Data
    #            ->
    # hemolyis_model/Data
    # --------------------------------------------------------

    try:

        expected_data_path.symlink_to(
            PFEATURE_DATA_DIR,
            target_is_directory=True
        )

    except OSError as exc:

        raise RuntimeError(
            "\nCould not create HemoPI2 compatibility "
            "symbolic link.\n\n"
            f"Source:\n"
            f"  {PFEATURE_DATA_DIR}\n\n"
            f"Link:\n"
            f"  {expected_data_path}\n\n"
            f"Error:\n"
            f"  {exc}"
        ) from exc

    print(
        "\nCreated HemoPI2 compatibility path:"
    )

    print(
        f"{expected_data_path}"
    )

    print("->")

    print(
        f"{PFEATURE_DATA_DIR}"
    )


# ============================================================
# Sequence preprocessing
# ============================================================

def prepare_sequences(csv_file):
    """
    Read sequences from a CSV file and reproduce the
    sequence-length preprocessing used by HemoPI2.

    Returns
    -------
    original_sequences : np.ndarray
        Cleaned original sequences.

    processed_sequences : np.ndarray
        Sequences actually passed to the Pfeature extractor.
    """

    csv_file = Path(csv_file).resolve()

    # --------------------------------------------------------
    # Read dataset
    # --------------------------------------------------------

    df = pd.read_csv(csv_file)

    # --------------------------------------------------------
    # Check sequence column
    # --------------------------------------------------------

    if SEQUENCE_COLUMN not in df.columns:

        raise ValueError(
            f"\nColumn '{SEQUENCE_COLUMN}' not found in:\n"
            f"{csv_file}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Check missing sequences
    # --------------------------------------------------------

    missing_mask = df[SEQUENCE_COLUMN].isna()

    if missing_mask.any():

        bad_rows = (
            df.index[missing_mask]
            .tolist()
        )

        raise ValueError(
            "\nMissing sequences detected.\n\n"
            f"Dataset:\n"
            f"{csv_file}\n\n"
            f"Rows:\n"
            f"{bad_rows}"
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
    # Check empty sequences
    # --------------------------------------------------------

    empty_mask = (
        sequences.str.len() == 0
    )

    if empty_mask.any():

        bad_rows = (
            df.index[empty_mask]
            .tolist()
        )

        raise ValueError(
            "\nEmpty sequences detected.\n\n"
            f"Dataset:\n"
            f"{csv_file}\n\n"
            f"Rows:\n"
            f"{bad_rows}"
        )

    # --------------------------------------------------------
    # Validate amino acids
    # --------------------------------------------------------

    invalid_sequences = []

    for row_index, sequence in sequences.items():

        invalid_residues = (
            set(sequence) - VALID_AA
        )

        if invalid_residues:

            invalid_sequences.append(
                (
                    row_index,
                    sequence,
                    sorted(invalid_residues)
                )
            )

    if invalid_sequences:

        print(
            "\nInvalid amino-acid sequences detected:"
        )

        for (
            row_index,
            sequence,
            invalid_residues
        ) in invalid_sequences[:20]:

            print(
                f"  row={row_index} | "
                f"sequence={sequence} | "
                f"invalid={invalid_residues}"
            )

        raise ValueError(
            "\nSequences contain residues outside "
            "the 20 standard amino acids."
        )

    # --------------------------------------------------------
    # HemoPI2 preprocessing
    #
    # Original HemoPI2 truncates sequences longer than
    # 40 amino acids.
    # --------------------------------------------------------

    processed_sequences = (
        sequences.str[:MAX_LENGTH]
    )

    n_truncated = int(
        (
            sequences.str.len()
            >
            MAX_LENGTH
        ).sum()
    )

    # --------------------------------------------------------
    # Check minimum length
    # --------------------------------------------------------

    short_mask = (
        processed_sequences.str.len() < 3
    )

    if short_mask.any():

        bad_rows = (
            df.index[short_mask]
            .tolist()
        )

        raise ValueError(
            "\nSequences shorter than 3 amino acids "
            "were detected.\n\n"
            f"Rows:\n"
            f"{bad_rows}"
        )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    lengths = sequences.str.len()

    print("\nDataset:")
    print(csv_file)

    print(
        f"Sequences        : {len(sequences)}"
    )

    print(
        f"Length range     : "
        f"{lengths.min()}-{lengths.max()} aa"
    )

    print(
        f"Truncated >40 aa : {n_truncated}"
    )

    # --------------------------------------------------------
    # Return aligned arrays
    # --------------------------------------------------------

    return (
        sequences.to_numpy(dtype=str),
        processed_sequences.to_numpy(dtype=str)
    )


# ============================================================
# Validate generated feature matrix
# ============================================================
def validate_feature_matrix(
    pfeature_df,
    original_sequences,
    dataset_name
):
    """
    Validate the Pfeature matrix generated by HemoPI2.

    Parameters
    ----------
    pfeature_df : pandas.DataFrame
        Raw feature dataframe produced by
        composition_calculate_hemopi2_2.py.

    original_sequences : array-like
        Original peptide sequences.

    dataset_name : str
        Name of the dataset being processed.

    Returns
    -------
    numeric_df : pandas.DataFrame
        Numeric feature dataframe.

    pfeatures : numpy.ndarray
        Feature matrix with dtype float32.

    feature_names : numpy.ndarray
        Feature names stored as Unicode strings rather than
        Python objects. This allows safe NPZ loading with
        allow_pickle=False.
    """

    # ========================================================
    # Remove accidental index columns
    # ========================================================

    pfeature_df = pfeature_df.loc[
        :,
        ~pfeature_df.columns
        .astype(str)
        .str.startswith("Unnamed:")
    ]

    # ========================================================
    # Check number of feature rows
    # ========================================================

    if len(pfeature_df) != len(original_sequences):

        raise RuntimeError(
            "\nFeature/sequence count mismatch.\n\n"
            f"Dataset       : {dataset_name}\n"
            f"Sequences     : {len(original_sequences)}\n"
            f"Feature rows  : {len(pfeature_df)}"
        )

    # ========================================================
    # Convert all features to numeric
    # ========================================================

    numeric_df = pfeature_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ========================================================
    # Detect values that became NaN only because conversion
    # failed.
    # ========================================================

    conversion_problem_columns = []

    for column in pfeature_df.columns:

        original_nan = (
            pfeature_df[column].isna()
        )

        converted_nan = (
            numeric_df[column].isna()
        )

        newly_created_nan = (
            converted_nan &
            ~original_nan
        )

        if newly_created_nan.any():

            conversion_problem_columns.append(
                str(column)
            )

    if conversion_problem_columns:

        raise ValueError(
            "\nNon-numeric values detected in "
            "Pfeature output.\n\n"
            f"Problematic columns:\n"
            f"{conversion_problem_columns[:30]}"
        )

    # ========================================================
    # Convert feature matrix to float32
    # ========================================================

    pfeatures = np.asarray(
        numeric_df.to_numpy(),
        dtype=np.float32
    )

    # ========================================================
    # IMPORTANT FIX
    #
    # Do NOT allow feature_names to become dtype=object.
    #
    # dtype=str creates a NumPy Unicode array such as:
    #
    #     dtype='<U20'
    #
    # instead of:
    #
    #     dtype=object
    #
    # Therefore it can safely be loaded using:
    #
    #     allow_pickle=False
    # ========================================================

    feature_names = np.asarray(
        [
            str(column)
            for column in numeric_df.columns
        ],
        dtype=str
    )

    # ========================================================
    # Check NaN
    # ========================================================

    nan_mask = np.isnan(pfeatures)

    nan_count = int(
        nan_mask.sum()
    )

    if nan_count > 0:

        print(
            f"\nWARNING: {nan_count} NaN "
            "values detected in Pfeature matrix."
        )

        rows, cols = np.where(
            nan_mask
        )

        print(
            "\nFirst affected positions:"
        )

        for row, col in zip(
            rows[:20],
            cols[:20]
        ):

            print(
                f"  row={row} | "
                f"sequence={original_sequences[row]} | "
                f"feature={feature_names[col]}"
            )

    # ========================================================
    # Check infinity
    # ========================================================

    inf_mask = np.isinf(
        pfeatures
    )

    inf_count = int(
        inf_mask.sum()
    )

    if inf_count > 0:

        print(
            f"\nERROR: {inf_count} infinite "
            "values detected in Pfeature matrix."
        )

        rows, cols = np.where(
            inf_mask
        )

        print(
            "\nFirst affected positions:"
        )

        for row, col in zip(
            rows[:20],
            cols[:20]
        ):

            print(
                f"  row={row} | "
                f"sequence={original_sequences[row]} | "
                f"feature={feature_names[col]}"
            )

        raise ValueError(
            "\nInfinite values detected in "
            "Pfeature matrix."
        )

    # ========================================================
    # Final validation report
    # ========================================================

    print(
        f"\nValidated Pfeature matrix:"
    )

    print(
        f"  Samples       : "
        f"{pfeatures.shape[0]}"
    )

    print(
        f"  Features      : "
        f"{pfeatures.shape[1]}"
    )

    print(
        f"  Feature dtype : "
        f"{pfeatures.dtype}"
    )

    print(
        f"  Name dtype    : "
        f"{feature_names.dtype}"
    )

    print(
        f"  NaN values    : "
        f"{nan_count}"
    )

    print(
        f"  Inf values    : "
        f"{inf_count}"
    )

    return (
        numeric_df,
        pfeatures,
        feature_names
    )



# ============================================================
# Compute Pfeatures
# ============================================================
def compute_pfeatures(csv_file, dataset_name):
    """
    Compute HemoPI2 Pfeatures for one dataset.

    Parameters
    ----------
    csv_file : str or Path
        Path to the input CSV containing peptide sequences.

    dataset_name : str
        Name used for output files, e.g.
        "crossvalidation" or "independent_test".

    Returns
    -------
    Path
        Path to the generated compressed NPZ file.
    """

    # ============================================================
    # Resolve input path
    # ============================================================

    csv_file = Path(csv_file).resolve()

    # ============================================================
    # Prepare sequences
    # ============================================================

    (
        original_sequences,
        processed_sequences
    ) = prepare_sequences(csv_file)

    # ============================================================
    # Create output directory
    # ============================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ============================================================
    # Create temporary directory
    #
    # Absolute paths are important because the HemoPI2 subprocess
    # runs with cwd=HEMOLYSIS_MODEL_DIR.
    # ============================================================

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"pfeature_{dataset_name}_",
            dir=str(BASE_DIR)
        )
    ).resolve()

    sequence_file = (
        temp_dir /
        "sequences.txt"
    ).resolve()

    raw_feature_file = (
        temp_dir /
        "pfeatures.csv"
    ).resolve()

    try:

        # ========================================================
        # Write one processed sequence per line
        # ========================================================

        pd.Series(
            processed_sequences
        ).to_csv(
            sequence_file,
            index=False,
            header=False
        )

        # ========================================================
        # Report
        # ========================================================

        print("\n" + "=" * 70)

        print(
            f"COMPUTING PFEATURES: "
            f"{dataset_name.upper()}"
        )

        print("=" * 70)

        print(
            f"Input dataset : {csv_file}"
        )

        print(
            f"Sequence file : {sequence_file}"
        )

        print(
            f"Number seqs   : {len(processed_sequences)}"
        )

        # ========================================================
        # Construct HemoPI2 command
        # ========================================================

        command = [
            sys.executable,
            str(PFEATURE_SCRIPT),
            str(sequence_file),
            str(temp_dir),
            str(raw_feature_file)
        ]

        # ========================================================
        # Run original HemoPI2 extractor
        #
        # cwd is deliberately set to HEMOLYSIS_MODEL_DIR.
        #
        # The original extractor uses paths such as:
        #
        #     Model/Data/PhysicoChemical.csv
        #
        # Our compatibility link makes this resolve to the actual
        # hemolyis_model/Data directory.
        # ========================================================

        print(
            "\nRunning original HemoPI2 "
            "Pfeature extractor..."
        )

        try:

            subprocess.run(
                command,
                check=True,
                cwd=str(HEMOLYSIS_MODEL_DIR)
            )

        except subprocess.CalledProcessError as exc:

            raise RuntimeError(
                "\nHemoPI2 Pfeature extraction failed."
                "\n\n"
                f"Dataset:\n"
                f"  {csv_file}\n\n"
                f"Working directory:\n"
                f"  {HEMOLYSIS_MODEL_DIR}\n\n"
                f"Command:\n"
                f"  {' '.join(command)}\n\n"
                f"Exit code:\n"
                f"  {exc.returncode}"
            ) from exc

        # ========================================================
        # Verify that HemoPI2 generated the output
        # ========================================================

        if not raw_feature_file.exists():

            raise RuntimeError(
                "\nHemoPI2 extractor completed but "
                "did not create the expected output:\n"
                f"{raw_feature_file}"
            )

        # ========================================================
        # Load generated Pfeatures
        # ========================================================

        print(
            "\nLoading generated Pfeatures..."
        )

        pfeature_df = pd.read_csv(
            raw_feature_file
        )

        # ========================================================
        # Validate generated feature matrix
        # ========================================================

        (
            pfeature_df,
            pfeatures,
            feature_names
        ) = validate_feature_matrix(
            pfeature_df=pfeature_df,
            original_sequences=original_sequences,
            dataset_name=dataset_name
        )

        # ========================================================
        # Prepare arrays for NPZ storage
        # ========================================================
        #
        # IMPORTANT:
        #
        # All strings are explicitly converted to NumPy Unicode
        # arrays. This prevents dtype=object and allows the saved
        # NPZ file to be loaded safely with:
        #
        #     allow_pickle=False
        #
        # ========================================================

        original_sequences_np = np.asarray(
            [
                str(sequence)
                for sequence in original_sequences
            ],
            dtype=str
        )

        processed_sequences_np = np.asarray(
            [
                str(sequence)
                for sequence in processed_sequences
            ],
            dtype=str
        )

        feature_names_np = np.asarray(
            [
                str(name)
                for name in feature_names
            ],
            dtype=str
        )

        pfeatures_np = np.asarray(
            pfeatures,
            dtype=np.float32
        )

        # ========================================================
        # Safety check: do not save object arrays
        # ========================================================

        arrays_to_save = {
            "Sequence":
                original_sequences_np,

            "ProcessedSequence":
                processed_sequences_np,

            "Pfeature":
                pfeatures_np,

            "feature_names":
                feature_names_np,
        }

        for (
            array_name,
            array_value
        ) in arrays_to_save.items():

            if array_value.dtype == object:

                raise TypeError(
                    f"\nRefusing to save '{array_name}' "
                    "because it has dtype=object.\n"
                    "All NPZ arrays must use explicit "
                    "NumPy dtypes."
                )

        # ========================================================
        # Final consistency checks before saving
        # ========================================================

        n_sequences = len(
            original_sequences_np
        )

        n_processed_sequences = len(
            processed_sequences_np
        )

        n_feature_rows = (
            pfeatures_np.shape[0]
        )

        n_features = (
            pfeatures_np.shape[1]
        )

        n_feature_names = len(
            feature_names_np
        )

        # --------------------------------------------------------
        # Sequence / feature-row alignment
        # --------------------------------------------------------

        if not (
            n_sequences
            ==
            n_processed_sequences
            ==
            n_feature_rows
        ):

            raise RuntimeError(
                "\nSequence/Pfeature alignment error "
                "before saving.\n\n"
                f"Original sequences  : "
                f"{n_sequences}\n"
                f"Processed sequences : "
                f"{n_processed_sequences}\n"
                f"Pfeature rows       : "
                f"{n_feature_rows}"
            )

        # --------------------------------------------------------
        # Feature-name / feature-column alignment
        # --------------------------------------------------------

        if (
            n_features
            !=
            n_feature_names
        ):

            raise RuntimeError(
                "\nFeature-name alignment error "
                "before saving.\n\n"
                f"Pfeature columns : "
                f"{n_features}\n"
                f"Feature names    : "
                f"{n_feature_names}"
            )

        # --------------------------------------------------------
        # Check NaN
        # --------------------------------------------------------

        nan_count = int(
            np.isnan(
                pfeatures_np
            ).sum()
        )

        if nan_count > 0:

            raise ValueError(
                "\nCannot save Pfeature matrix: "
                f"{nan_count} NaN values detected."
            )

        # --------------------------------------------------------
        # Check Inf
        # --------------------------------------------------------

        inf_count = int(
            np.isinf(
                pfeatures_np
            ).sum()
        )

        if inf_count > 0:

            raise ValueError(
                "\nCannot save Pfeature matrix: "
                f"{inf_count} infinite values detected."
            )

        # ========================================================
        # Define final NPZ output
        # ========================================================

        npz_output = (
            OUTPUT_DIR /
            f"{dataset_name}_pfeatures.npz"
        )

        # ========================================================
        # Save NPZ
        # ========================================================

        np.savez_compressed(
            npz_output,

            # Original cleaned sequence
            Sequence=original_sequences_np,

            # Sequence actually supplied to HemoPI2
            ProcessedSequence=processed_sequences_np,

            # Feature matrix:
            # shape = (number of peptides, 1167)
            Pfeature=pfeatures_np,

            # Names of the 1167 descriptors
            feature_names=feature_names_np
        )

        # ========================================================
        # Save human-readable inspection CSV
        # ========================================================

        inspection_df = (
            pfeature_df.copy()
        )

        inspection_df.insert(
            0,
            "ProcessedSequence",
            processed_sequences_np
        )

        inspection_df.insert(
            0,
            "Sequence",
            original_sequences_np
        )

        csv_output = (
            OUTPUT_DIR /
            f"{dataset_name}_pfeatures.csv"
        )

        inspection_df.to_csv(
            csv_output,
            index=False
        )

        # ========================================================
        # Final report
        # ========================================================

        print(
            "\nPFEATURE EXTRACTION SUCCESSFUL"
        )

        print(
            f"Dataset            : "
            f"{dataset_name}"
        )

        print(
            f"Sequences          : "
            f"{n_sequences}"
        )

        print(
            f"Feature matrix     : "
            f"{pfeatures_np.shape}"
        )

        print(
            f"Number features    : "
            f"{n_features}"
        )

        print(
            f"NaN values         : "
            f"{nan_count}"
        )

        print(
            f"Inf values         : "
            f"{inf_count}"
        )

        print(
            f"Sequence dtype     : "
            f"{original_sequences_np.dtype}"
        )

        print(
            f"Processed dtype    : "
            f"{processed_sequences_np.dtype}"
        )

        print(
            f"Pfeature dtype     : "
            f"{pfeatures_np.dtype}"
        )

        print(
            f"Feature name dtype : "
            f"{feature_names_np.dtype}"
        )

        print(
            f"\nNPZ output:\n"
            f"{npz_output}"
        )

        print(
            f"\nInspection CSV:\n"
            f"{csv_output}"
        )

        # ========================================================
        # Return generated NPZ
        # ========================================================

        return npz_output

    finally:

        # ========================================================
        # Always remove temporary files
        # ========================================================

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# ============================================================
# Verify generated NPZ
# ============================================================

def verify_npz(npz_file):
    """
    Reload a generated NPZ file and verify:

    1. Required arrays exist.
    2. No object arrays are required.
    3. Sequence rows align with feature rows.
    4. Number of feature names matches feature columns.
    5. Pfeature matrix is numeric.
    6. No NaN values exist.
    7. No infinite values exist.
    """

    npz_file = Path(
        npz_file
    ).resolve()

    print("\n" + "-" * 70)
    print("VERIFYING NPZ")
    print("-" * 70)

    print(
        f"File:\n{npz_file}"
    )

    # ========================================================
    # IMPORTANT:
    #
    # Keep allow_pickle=False.
    #
    # We deliberately save strings as Unicode arrays rather
    # than Python object arrays.
    # ========================================================

    with np.load(
        npz_file,
        allow_pickle=False
    ) as data:

        # ====================================================
        # Required keys
        # ====================================================

        required_keys = {
            "Sequence",
            "ProcessedSequence",
            "Pfeature",
            "feature_names",
        }

        available_keys = set(
            data.files
        )

        missing_keys = (
            required_keys -
            available_keys
        )

        if missing_keys:

            raise RuntimeError(
                "\nSaved NPZ is missing "
                "required arrays:\n"
                f"{sorted(missing_keys)}"
            )

        # ====================================================
        # Load arrays
        # ====================================================

        sequences = (
            data["Sequence"]
        )

        processed_sequences = (
            data["ProcessedSequence"]
        )

        pfeatures = (
            data["Pfeature"]
        )

        feature_names = (
            data["feature_names"]
        )

        # ====================================================
        # Check object dtypes
        # ====================================================

        arrays = {
            "Sequence":
                sequences,

            "ProcessedSequence":
                processed_sequences,

            "Pfeature":
                pfeatures,

            "feature_names":
                feature_names,
        }

        for (
            array_name,
            array_value
        ) in arrays.items():

            if array_value.dtype == object:

                raise RuntimeError(
                    f"\nNPZ verification failed.\n"
                    f"'{array_name}' has "
                    "dtype=object."
                )

        # ====================================================
        # Check dimensions
        # ====================================================

        if sequences.ndim != 1:

            raise RuntimeError(
                "\nSequence array must be "
                "one-dimensional.\n"
                f"Found shape: "
                f"{sequences.shape}"
            )

        if processed_sequences.ndim != 1:

            raise RuntimeError(
                "\nProcessedSequence array must "
                "be one-dimensional.\n"
                f"Found shape: "
                f"{processed_sequences.shape}"
            )

        if pfeatures.ndim != 2:

            raise RuntimeError(
                "\nPfeature matrix must be "
                "two-dimensional.\n"
                f"Found shape: "
                f"{pfeatures.shape}"
            )

        if feature_names.ndim != 1:

            raise RuntimeError(
                "\nfeature_names must be "
                "one-dimensional.\n"
                f"Found shape: "
                f"{feature_names.shape}"
            )

        # ====================================================
        # Check row alignment
        # ====================================================

        n_sequences = len(
            sequences
        )

        n_processed = len(
            processed_sequences
        )

        n_feature_rows = (
            pfeatures.shape[0]
        )

        if not (
            n_sequences
            ==
            n_processed
            ==
            n_feature_rows
        ):

            raise RuntimeError(
                "\nNPZ row-alignment failure.\n\n"
                f"Sequence rows          : "
                f"{n_sequences}\n"
                f"ProcessedSequence rows : "
                f"{n_processed}\n"
                f"Pfeature rows          : "
                f"{n_feature_rows}"
            )

        # ====================================================
        # Check feature-name alignment
        # ====================================================

        n_features = (
            pfeatures.shape[1]
        )

        n_feature_names = len(
            feature_names
        )

        if (
            n_features
            !=
            n_feature_names
        ):

            raise RuntimeError(
                "\nFeature-name alignment "
                "failure.\n\n"
                f"Pfeature columns : "
                f"{n_features}\n"
                f"Feature names    : "
                f"{n_feature_names}"
            )

        # ====================================================
        # Ensure Pfeature is numeric
        # ====================================================

        if not np.issubdtype(
            pfeatures.dtype,
            np.number
        ):

            raise RuntimeError(
                "\nPfeature matrix is not "
                "numeric.\n"
                f"dtype={pfeatures.dtype}"
            )

        # ====================================================
        # Check NaN
        # ====================================================

        nan_count = int(
            np.isnan(
                pfeatures
            ).sum()
        )

        if nan_count > 0:

            raise RuntimeError(
                "\nNPZ verification failed: "
                f"{nan_count} NaN values "
                "detected."
            )

        # ====================================================
        # Check infinity
        # ====================================================

        inf_count = int(
            np.isinf(
                pfeatures
            ).sum()
        )

        if inf_count > 0:

            raise RuntimeError(
                "\nNPZ verification failed: "
                f"{inf_count} infinite values "
                "detected."
            )

        # ====================================================
        # Check empty feature matrix
        # ====================================================

        if n_features == 0:

            raise RuntimeError(
                "\nPfeature matrix contains "
                "zero features."
            )

        # ====================================================
        # Report
        # ====================================================

        print(
            "\nNPZ verification passed."
        )

        print(
            f"  Sequence          : "
            f"{sequences.shape} | "
            f"dtype={sequences.dtype}"
        )

        print(
            f"  ProcessedSequence : "
            f"{processed_sequences.shape} | "
            f"dtype={processed_sequences.dtype}"
        )

        print(
            f"  Pfeature          : "
            f"{pfeatures.shape} | "
            f"dtype={pfeatures.dtype}"
        )

        print(
            f"  feature_names     : "
            f"{feature_names.shape} | "
            f"dtype={feature_names.dtype}"
        )

        print(
            f"  NaN values        : "
            f"{nan_count}"
        )

        print(
            f"  Inf values        : "
            f"{inf_count}"
        )

        print(
            f"  Samples           : "
            f"{n_sequences}"
        )

        print(
            f"  Features          : "
            f"{n_features}"
        )

    print("-" * 70)

    return True


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PFEATURE EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate HemoPI2 files
    # --------------------------------------------------------

    validate_pfeature_environment()

    # --------------------------------------------------------
    # Create compatibility path:
    #
    # hemolyis_model/Model/Data
    #           ->
    # hemolyis_model/Data
    #
    # so the original HemoPI2 code does not need modification.
    # --------------------------------------------------------

    prepare_hemopi2_path_compatibility()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process datasets
    # --------------------------------------------------------

    generated_files = {}

    for (
        dataset_name,
        csv_file
    ) in DATASETS.items():

        if not csv_file.exists():

            raise FileNotFoundError(
                "\nDataset not found:\n"
                f"{csv_file}"
            )

        output_file = compute_pfeatures(
            csv_file=csv_file,
            dataset_name=dataset_name
        )

        generated_files[
            dataset_name
        ] = output_file

        # Verify immediately
        verify_npz(
            output_file
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "ALL PFEATURES GENERATED SUCCESSFULLY"
    )

    print("=" * 70)

    for (
        dataset_name,
        output_file
    ) in generated_files.items():

        print(
            f"{dataset_name:20s}: "
            f"{output_file}"
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()