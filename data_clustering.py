import subprocess
import os
import io
import tempfile
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

def run_cdhit(sequences, identity_threshold=0.60, word_size=4):
    """
    Writes sequences to FASTA, executes CD-HIT using isolated temporary files,
    and returns a cluster mapping dictionary {seq_id: cluster_id}.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        fasta_path = os.path.join(temp_dir, "input.fasta")
        output_prefix = os.path.join(temp_dir, "cdhit_out")
        clstr_path = f"{output_prefix}.clstr"

        with open(fasta_path, "w") as f:
            for idx, seq in enumerate(sequences):
                f.write(f">seq_{idx}\n{seq}\n")

        cmd = [
            "cd-hit",
            "-i", fasta_path,
            "-o", output_prefix,
            "-c", str(identity_threshold),
            "-n", str(word_size),
            "-d", "0",
            "-sc", "1"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        seq_to_cluster = {}
        current_cluster = None

        with open(clstr_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">Cluster"):
                    current_cluster = int(line.split()[1])
                elif line:
                    seq_id = line.split(">")[1].split("...")[0]
                    seq_to_cluster[seq_id] = current_cluster

        return seq_to_cluster

def build_homology_aware_splits(
    df,
    seq_col="sequence",
    identity_threshold=0.60,
    word_size=4,
    split_mode="ratio",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    n_train=None,
    n_val=None,
    n_test=None,
    n_cv_folds=5,
    random_state=42,
):
    """Clusters sequences and delegates data partitioning to

    split_dataframe_by_clusters.
    """
    df = df.copy().reset_index(drop=True)

    # 1. Cluster peptides with CD-HIT
    cluster_map = run_cdhit(
        df[seq_col].values,
        identity_threshold=identity_threshold,
        word_size=word_size,
    )
    max_cluster = max(cluster_map.values()) if cluster_map else 0
    cluster_ids = []

    for i in range(len(df)):
        seq_key = f"seq_{i}"
        if seq_key in cluster_map:
            cluster_ids.append(cluster_map[seq_key])
        else:
            max_cluster += 1
            cluster_ids.append(max_cluster)

    df["cluster_id"] = cluster_ids

    # 2. Split using extracted method
    return split_dataframe_by_clusters(
        df=df,
        group_col="cluster_id",
        split_mode=split_mode,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        n_cv_folds=n_cv_folds,
        random_state=random_state,
    )

def split_dataframe_by_clusters(
    df,
    group_col="cluster_id",
    split_mode="ratio",  # "ratio" or "count"
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    n_train=None,
    n_val=None,
    n_test=None,
    n_cv_folds=5,
    random_state=42,
):
    """
    Partitions a DataFrame to match target SAMPLE (row) counts as closely as possible,
    while guaranteeing that all rows in any given cluster stay strictly in one split.
    """
    np.random.seed(random_state)
    df = df.copy()
    total_samples = len(df)

    # 1. Calculate target SAMPLE (row) counts
    if split_mode == "count":
        target_train = n_train if n_train is not None else 0
        target_val = n_val if n_val is not None else 0
        target_test = n_test if n_test is not None else 0
        
        if (target_train + target_val + target_test) > total_samples:
            raise ValueError(
                f"Requested samples ({target_train + target_val + target_test}) "
                f"exceed total dataset rows ({total_samples})."
            )
    elif split_mode == "ratio":
        target_test = int(round(total_samples * test_ratio))
        target_val = int(round(total_samples * val_ratio))
        target_train = total_samples - target_test - target_val
    else:
        raise ValueError(f"Invalid split_mode '{split_mode}'. Use 'ratio' or 'count'.")

    # 2. Get sample count per cluster
    cluster_counts = df[group_col].value_counts().to_dict()
    unique_clusters = list(cluster_counts.keys())
    np.random.shuffle(unique_clusters)

    # 3. Greedily assign WHOLE clusters to splits based on row deficits
    train_clusters, val_clusters, test_clusters = set(), set(), set()
    current_train, current_val, current_test = 0, 0, 0

    for cid in unique_clusters:
        c_size = cluster_counts[cid]

        # Calculate sample deficit for each split
        deficits = [
            (target_test - current_test, "test"),
            (target_val - current_val, "val"),
            (target_train - current_train, "train")
        ]
        # Sort by largest remaining sample need
        deficits.sort(key=lambda x: x[0], reverse=True)
        best_split = deficits[0][1]

        if best_split == "test" and (current_test < target_test or split_mode == "ratio"):
            test_clusters.add(cid)
            current_test += c_size
        elif best_split == "val" and (current_val < target_val or split_mode == "ratio"):
            val_clusters.add(cid)
            current_val += c_size
        else:
            if split_mode == "count" and current_train >= target_train and deficits[0][0] <= 0:
                # Count mode limit reached; leave extra clusters unassigned
                pass
            else:
                train_clusters.add(cid)
                current_train += c_size

    def assign_split(cid):
        if cid in test_clusters:
            return "test"
        elif cid in val_clusters:
            return "val"
        elif cid in train_clusters:
            return "train"
        return "unused"

    df["split"] = df[group_col].apply(assign_split)

    # 4. GroupKFold CV on the training samples
    df["cv_fold"] = -1
    train_idx = df[df["split"] == "train"].index
    train_df = df.loc[train_idx]

    num_train_clusters = train_df[group_col].nunique()
    if num_train_clusters >= 2:
        actual_folds = min(n_cv_folds, num_train_clusters)
        gkf = GroupKFold(n_splits=actual_folds)

        for fold, (_, fold_val_idx) in enumerate(
            gkf.split(train_df, groups=train_df[group_col])
        ):
            df.loc[train_idx[fold_val_idx], "cv_fold"] = fold

    return df

if __name__ == "__main__":
    csv_path = "data/ecoli_dataset.csv"  # Update with the actual path to your CSV file
    df = pd.read_csv(csv_path)
    processed_df = build_homology_aware_splits(df, split_mode="count", n_train=3259, n_val=815, n_test=719, n_cv_folds=5)
    print(processed_df[["sequence", "cluster_id", "split", "cv_fold"]])
    
    # 2. Export the DataFrame to CSV
    output_filename = "data/clustered_dataset_splits.csv"
    processed_df.to_csv(output_filename, index=False)

    print(f"Dataset successfully saved to: {output_filename}")