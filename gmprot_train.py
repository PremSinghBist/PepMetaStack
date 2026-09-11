#!/usr/bin/env python3
from dataclasses import asdict
import os
from pathlib import Path
import random

import numpy as np
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import r2_score
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import data_util
import data_visualization
from experimental_config import ExperimentConfig

# --------------------------- Setup & Config ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

cfg = ExperimentConfig()
EMB_DIM = 1024
PHYSIO_DIM = (
    cfg.physio_feature_dim + cfg.blosum_feature_dim + cfg.sinusoidal_feature_dim
)

# Global Amino Acid Lookup Table (Instantiated ONCE to prevent graph/memory bloat)
AA_LOOKUP_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list("ACDEFGHIKLMNPQRSTVWY")),
        values=tf.constant(list(range(1, 21)), dtype=tf.int32),
    ),
    default_value=0,
)


def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class SequenceCNN(layers.Layer):
    def __init__(self, output_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.embed = layers.Embedding(
            cfg.cnn_input_dim, cfg.cnn_emb_output_dim, mask_zero=True
        )

        self.conv3 = layers.Conv1D(cfg.cnn_filters, 3, padding="same", activation="relu")
        self.conv5 = layers.Conv1D(cfg.cnn_filters, 5, padding="same", activation="relu")
        self.conv7 = layers.Conv1D(cfg.cnn_filters, 7, padding="same", activation="relu")
        self.conv11 = layers.Conv1D(cfg.cnn_filters, 11, padding="same", activation="relu")

        self.pool = layers.GlobalMaxPooling1D()
        self.proj = layers.Dense(self.output_dim, activation="relu")

    def call(self, seq_ids):
        x = self.embed(seq_ids)
        mask = tf.cast(tf.not_equal(seq_ids, 0), tf.float32)
        mask = tf.expand_dims(mask, -1)
        x = x * mask

        def masked_pool(conv_out):
            large_negative = -1e9
            conv_out_masked = conv_out + (1.0 - mask) * large_negative
            return self.pool(conv_out_masked)

        c3 = masked_pool(self.conv3(x))
        c5 = masked_pool(self.conv5(x))
        c7 = masked_pool(self.conv7(x))
        c11 = masked_pool(self.conv11(x))

        x = tf.concat([c3, c5, c7, c11], axis=-1)
        return self.proj(x)


@tf.keras.utils.register_keras_serializable()
class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self,
        num_heads=8,
        embed_dim=256,
        dense_dim=512,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dropout = dropout

        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(
                dense_dim,
                activation=cfg.ffn_A,
                kernel_initializer=keras.initializers.GlorotUniform(seed=101), # type: ignore
            ),
            layers.Dense(
                embed_dim,
                kernel_initializer=keras.initializers.GlorotUniform(seed=102), # type: ignore
            ),
        ])

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(dropout, seed=120)
        self.drop2 = layers.Dropout(dropout, seed=121)

    def call(self, x, training=False):
        x = x[:, tf.newaxis, :]
        attn_out = self.self_attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))

        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return tf.squeeze(x, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "dropout": self.dropout,
        })
        return config


# --------------------------- Data Pipeline ---------------------------
def generator(sample_list):
    for item in sample_list:
        emb = item["emb"]
        if emb.ndim == 2:
            emb = np.mean(emb, axis=0)

        physio_concat = np.concatenate(
            [item["physio"], item["blosum"], item["sino"]], axis=0
        ).astype(np.float32)

        yield (
            emb.astype(np.float32),
            physio_concat,
            item["seq"],
            np.array([item["mic"]], dtype=np.float32),
        )


def prepare_batch(batched_emb, batched_physio, batched_seq, batched_labels):
    seq_strings = tf.strings.strip(batched_seq)
    chars = tf.strings.unicode_split(seq_strings, "UTF-8")

    # Uses the global static lookup table
    seq_ids_padded = AA_LOOKUP_TABLE.lookup(chars).to_tensor(default_value=0)

    labels = tf.cast(batched_labels, tf.float32)
    if len(labels.shape) == 1:
        labels = tf.expand_dims(labels, axis=-1)

    inputs = {
        "seq_emb": tf.cast(batched_emb, tf.float32),
        "physio_features": tf.cast(batched_physio, tf.float32),
        "seq_ids": seq_ids_padded,
    }
    return inputs, labels


def make_dataset(sample_list, shuffle=False):
    emb_spec = tf.TensorSpec(shape=[EMB_DIM], dtype=tf.float32)
    physio_spec = tf.TensorSpec(shape=[PHYSIO_DIM], dtype=tf.float32)
    seq_spec = tf.TensorSpec(shape=(), dtype=tf.string)
    label_spec = tf.TensorSpec(shape=(1,), dtype=tf.float32)

    ds = tf.data.Dataset.from_generator(
        lambda: generator(sample_list),
        output_signature=(emb_spec, physio_spec, seq_spec, label_spec),
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)

    return (
        ds.batch(cfg.batch_size)
        .map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


# --------------------------- Model Architecture ---------------------------
class GMProt(keras.Model):

    def __init__(self, cfg: ExperimentConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        # Sequence Embedding Branch
        self.seq_proj = layers.Dense(
            cfg.seq_dense1, activation=cfg.seq_proj_A
        )
        self.seq_transformer = TransformerEncoderReadout(
            num_heads=cfg.transformer_heads,
            embed_dim=cfg.seq_dense1,
            dense_dim=cfg.transformer_ff_dim,
        )
        self.seq_out = layers.Dense(cfg.seq_dense2, activation=cfg.seq_out_A)
        self.seq_bottleneck = layers.Dense(
            cfg.seq_bottleneck_dim, activation=cfg.seq_bottleneck_A
        )

        # Physicochemical Branch
        self.physio_proj = layers.Dense(
            cfg.physio_proj, activation=cfg.physio_proj_A
        )
        self.physio_bottleneck = layers.Dense(
            cfg.physio_bottleneck_dim, activation=cfg.physio_bottleneck_A
        )

        # Sequence CNN Branch
        self.seq_cnn = SequenceCNN(output_dim=cfg.seq_bottleneck_dim)
        self.seq_cnn_norm = layers.LayerNormalization()
        self.seq_cnn_dropout = layers.Dropout(cfg.fuse_dropout)

        # Output Fusion Projection
        self.fusion_proj = layers.Dense(128, activation="relu")
        self.out = layers.Dense(1)

    def get_config(self):
        config = super().get_config()
        config.update({"cfg": asdict(self.cfg)})
        return config

    @classmethod
    def from_config(cls, config):
        cfg_dict = config.pop("cfg")
        return cls(ExperimentConfig(**cfg_dict), **config)
    

    # ======================================================================
    # Get 128-D embedding from fusion_proj
    # ======================================================================

    def get_fusion_embedding(self, inputs):
        """
        Return the 128-dimensional representation produced by fusion_proj.
        """

        seq_emb = inputs["seq_emb"]
        physio = inputs["physio_features"]
        seq_ids = inputs["seq_ids"]

        # Sequence Transformer path
        seq = self.seq_proj(seq_emb)
        seq = self.seq_transformer(seq, training=False)
        seq = self.seq_out(seq)
        seq_feat = self.seq_bottleneck(seq)

        # Physicochemical path
        physio_feat = self.physio_proj(physio)
        physio_feat = self.physio_bottleneck(physio_feat)

        # Sequence CNN path
        seq_cnn_feat = self.seq_cnn(seq_ids)
        seq_cnn_feat = self.seq_cnn_norm(seq_cnn_feat)

        # CNN gated modulation
        seq_feat = seq_feat * (
            1 + self.cfg.cnn_gating_threshold * tf.tanh(seq_cnn_feat)
        )

        # Fuse features
        fused_feat = tf.concat(
            [seq_feat, physio_feat],
            axis=-1
        )

        # 128-D GMProt embedding
        return self.fusion_proj(
            fused_feat,
            training=False
        )


    # ======================================================================
    # Prepare inputs for one sequence
    # ======================================================================

    def _prepare_fusion_inputs(self, item, sequence):
        """
        Prepare the inputs required by get_fusion_embedding().
        """

        # ProtT5 embedding
        emb = np.asarray(
            item["emb"],
            dtype=np.float32
        )

        # Convert residue-level embedding to sequence-level embedding
        if emb.ndim == 2:
            emb = np.mean(emb, axis=0)

        if emb.ndim != 1:
            raise ValueError(
                f"Unexpected embedding shape for {sequence}: {emb.shape}"
            )

        # Physicochemical + BLOSUM + sinusoidal features
        physio_features = np.concatenate(
            [
                np.asarray(item["physio"], dtype=np.float32),
                np.asarray(item["blosum"], dtype=np.float32),
                np.asarray(item["sino"], dtype=np.float32),
            ],
            axis=0
        )

        # Convert sequence to amino-acid IDs
        seq_tensor = tf.constant(
            [sequence],
            dtype=tf.string
        )

        chars = tf.strings.unicode_split(
            seq_tensor,
            "UTF-8"
        )

        seq_ids = (
            AA_LOOKUP_TABLE
            .lookup(chars)
            .to_tensor(default_value=0)
        )

        return {
            "seq_emb": tf.convert_to_tensor(
                emb[np.newaxis, :],
                dtype=tf.float32
            ),
            "physio_features": tf.convert_to_tensor(
                physio_features[np.newaxis, :],
                dtype=tf.float32
            ),
            "seq_ids": seq_ids,
        }


    # ======================================================================
    # Extract GMProt fusion embeddings for all sequences
    # ======================================================================

    def compute_save_fusion_embeddings(self, data_splits, output_path=None):
        """
        Extract 128-D GMProt fusion embeddings for all sequences
        in train, validation and test splits.

        Returns
        -------
        dict
            {
                "SEQUENCE_1": np.ndarray(shape=(128,)),
                "SEQUENCE_2": np.ndarray(shape=(128,)),
                ...
            }
        """

        # ------------------------------------------------------------------
        # Build sequence -> feature lookup
        # ------------------------------------------------------------------

        feature_lookup = {}

        for split in ["train", "val", "test"]:
            for item in data_splits.get(split, []):
                sequence = str(
                    item["seq"]
                ).strip()

                if sequence not in feature_lookup:
                    feature_lookup[sequence] = item

        print(
            f"\nTotal unique sequences: "
            f"{len(feature_lookup)}"
        )

        # ------------------------------------------------------------------
        # Extract embeddings
        # ------------------------------------------------------------------

        embedding_dict = {}

        print(
            "\nExtracting GMProt fusion embeddings..."
        )

        total = len(feature_lookup)

        for i, (sequence, item) in enumerate(
            feature_lookup.items(),
            start=1
        ):

            # Prepare inputs
            inputs = self._prepare_fusion_inputs(
                item,
                sequence
            )

            # Get 128-D embedding
            embedding = self.get_fusion_embedding(
                inputs
            )

            # TensorFlow -> NumPy
            embedding = (
                embedding
                .numpy()
                .squeeze(0)
                .astype(np.float32)
            )

            # Check shape
            if embedding.shape != (128,):
                raise ValueError(
                    f"Unexpected embedding shape for "
                    f"{sequence}: {embedding.shape}"
                )

            embedding_dict[sequence] = embedding

            if i % 100 == 0 or i == total:
                print(
                    f"Processed {i}/{total}"
                )

        # ------------------------------------------------------------------
        # Save embeddings
        # ------------------------------------------------------------------
        output_path = Path(output_path)

        if output_path.suffix.lower() != ".npy":
            output_path = output_path.with_suffix(".npy")

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            output_path,
            embedding_dict,
            allow_pickle=True
        )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------

        print("\n" + "=" * 60)
        print(
            "GMProt embedding extraction completed."
        )
        print("=" * 60)
        print(
            f"Sequences processed : {len(embedding_dict)}"
        )
        print(
            "Embedding dimension : 128"
        )
        print(
            f"Output file         : {output_path}"
        )
        print("=" * 60)

        return embedding_dict

    def call(self, inputs, training=False):
        seq_emb = inputs["seq_emb"]
        physio = inputs["physio_features"]
        seq_ids = inputs["seq_ids"]

        # Sequence Transformer path
        seq = self.seq_proj(seq_emb)
        seq = self.seq_transformer(seq, training=training)
        seq = self.seq_out(seq)
        seq_feat = self.seq_bottleneck(seq)

        # Physicochemical path
        physio_feat = self.physio_proj(physio)
        physio_feat = self.physio_bottleneck(physio_feat)

        # CNN Path & Gated Modulation
        seq_cnn_feat = self.seq_cnn(seq_ids)
        seq_cnn_feat = self.seq_cnn_norm(seq_cnn_feat)
        seq_cnn_feat = self.seq_cnn_dropout(seq_cnn_feat, training=training)

        seq_feat = seq_feat * (
            1 + self.cfg.cnn_gating_threshold * tf.tanh(seq_cnn_feat)
        )

        # Concatenate branches
        fused_feat = tf.concat([seq_feat, physio_feat], axis=-1)
        fused_feat = self.fusion_proj(fused_feat)

        return self.out(fused_feat)


# --------------------------- Training & Evaluation ---------------------------
def compute_metrics(model, dataset, model_path=None, visualize=False):
    y_true, y_pred = [], []
    for inputs, labels in dataset:
        preds = model(inputs, training=False)
        y_true.extend(labels.numpy().flatten())
        y_pred.extend(preds.numpy().flatten())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    metrics = {
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": float(pearsonr(y_true, y_pred)[0]),
        "Kendall": float(kendalltau(y_true, y_pred)[0]),
    }

    return metrics


def train_model(data_splits, model_name="GMProt_Experiment"):
    print(
        f"\nTrain: {len(data_splits['train'])}"
        f" | Val: {len(data_splits['val'])}"
        f" | Test: {len(data_splits['test'])}"
    )

    if not data_splits["train"]:
        raise ValueError("Training set is empty.")

    if not data_splits["val"]:
        raise ValueError("Validation set is empty.")

    if not data_splits["test"]:
        raise ValueError("Test set is empty.")
    
    
    train_ds = make_dataset(data_splits["train"], shuffle=True)
    val_ds = make_dataset(data_splits["val"])
    test_ds = make_dataset(data_splits["test"])

    model = GMProt(cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    model_path = Path(f"./model/{model_name}")
    model_path.mkdir(parents=True, exist_ok=True)
    cfg.save_config(model_path / "config.json")

    callbacks = [
        EarlyStopping(
            monitor="val_rmse",
            mode="min",
            patience=cfg.patience,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            str(model_path),
            save_best_only=True,
            monitor="val_rmse",
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_rmse", factor=0.5, patience=cfg.patience, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=2,
    )



    test_metrics = compute_metrics(model, test_ds, model_path=model_path)
    print(f"Test metrics: {test_metrics}")

    return model, test_metrics

def evaluate_saved_model(
    model_dir="None",
    dataset_csv=cfg.ecoli_dataset_csv,
    embedding_file=cfg.ecoli_prott5_embedding_file,
    blosum_csv=cfg.ecoli_blosum_csv,
    physio_csv=cfg.ecoli_physio_csv,
    sinusoidal_csv=cfg.ecoli_sinusoidal_csv,
):
    """Loads a saved Keras model checkpoint and computes performance metrics on the test set."""
    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # 1. Load data and extract test split
    print("Loading dataset features for evaluation...")
    data_splits = data_util.load_features(
        normalize_features=True,
        embedding_file=embedding_file,
        blosum_csv=blosum_csv,
        physio_csv=physio_csv,
        sinusoidal_csv=sinusoidal_csv,
        dataset_csv=dataset_csv,
    )

    test_ds = make_dataset(data_splits["test"], shuffle=False)
    print(f"Test split loaded successfully ({len(data_splits['test'])} samples).")

    # 2. Load model with custom layers registered
    print(f"Loading model checkpoint from {model_path}...")
    custom_objects = {
        "GMProt": GMProt,
        "TransformerEncoderReadout": TransformerEncoderReadout,
        "SequenceCNN": SequenceCNN,
    }

    try:
        model = keras.models.load_model(
            str(model_path), custom_objects=custom_objects
        )
    except Exception as e:
        print(
            f"Full model load failed ({e}). Re-instantiating architecture and loading weights..."
        )
        model = GMProt(cfg)
        dummy_inputs, _ = next(iter(test_ds))
        model(dummy_inputs, training=False)  # Initialize weights
        model.load_weights(str(model_path))

    # 3. Compute and display test metrics
    print("\nComputing metrics on test set...")
    test_metrics = compute_metrics(model, test_ds, model_path=model_path)

    print("\nTest Set Metrics:")
    for metric_name, metric_val in test_metrics.items():
        print(f"  {metric_name}: {metric_val:.4f}")

    return test_metrics

def main(model_name="GMProt_Experiment", fold_index=None):
    set_global_seed(cfg.seed)


    data_splits = data_util.load_features(
        embedding_file=cfg.ecoli_prott5_embedding_file,
        blosum_csv=cfg.ecoli_blosum_csv,
        physio_csv=cfg.ecoli_physio_csv,
        sinusoidal_csv=cfg.ecoli_sinusoidal_csv,
        dataset_csv=cfg.ecoli_dataset_csv,
        fold_index=fold_index
    )
    

    print(
        f"Data loaded successfully | "
        f"Train: {len(data_splits['train'])} | "
        f"Val: {len(data_splits['val'])} | "
        f"Test: {len(data_splits['test'])}"
    )

      
    
    #Homology aware splitting could not perform due to insufficient data. Using original splits instead.
    #data_splits = data_util.merge_and_resplit_by_counts(data_splits, seed=cfg.seed)
    
    model_name = f"{model_name}_{fold_index}"
    print(f"\nStarting training run for {model_name}...")

    model, test_metrics = train_model(data_splits, model_name=model_name)
    print("\nTraining Complete.")
    print("Final Test Set Performance:")
    for metric_name, metric_val in test_metrics.items():
        print(f"  {metric_name}: {metric_val:.4f}")
        
    
    return test_metrics
    
    '''evaluate_saved_model(
        model_dir=f"./model/{model_name}",
        dataset_csv=cfg.ecoli_dataset_csv,
        embedding_file=cfg.ecoli_prott5_embedding_file,
        blosum_csv=cfg.ecoli_blosum_csv,
        physio_csv=cfg.ecoli_physio_csv,
        sinusoidal_csv=cfg.ecoli_sinusoidal_csv,
    )'''
    

def fused_embed_extraction():
    set_global_seed(cfg.seed)

    print("Loading real dataset features and computing splits...")

    data_splits = data_util.load_features(
        normalize_features=True,
        embedding_file=cfg.ecoli_prott5_embedding_file,
        blosum_csv=cfg.ecoli_blosum_csv,
        physio_csv=cfg.ecoli_physio_csv,
        sinusoidal_csv=cfg.ecoli_sinusoidal_csv,
        dataset_csv=cfg.ecoli_dataset_csv,
    )
    
    print(
        f"Data loaded successfully | "
        f"Train: {len(data_splits['train'])} | "
        f"Val: {len(data_splits['val'])} | "
        f"Test: {len(data_splits['test'])}"
    )

    model_name = "GMProt_Experiment_1"
    model_path = Path(f"./model/{model_name}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Load the trained model
    custom_objects = {
        "GMProt": GMProt,
        "TransformerEncoderReadout": TransformerEncoderReadout,
        "SequenceCNN": SequenceCNN,
    }
    model: GMProt = keras.models.load_model(str(model_path), custom_objects=custom_objects)

    # Extract fusion embeddings for all sequences
    embedding_dict = model.compute_save_fusion_embeddings(data_splits, output_path=f"./data/gmprot_ecoli_fusion_embeddings.npy")  
     

   


if __name__ == "__main__":
    models = [1, 2, 3, 4, 5]
    test_metrics_list = []
    for fold_index in models:
        metrics = main(model_name="GMProt_Experiment", fold_index=fold_index)
        test_metrics_list.append(metrics)
        
    data_util.summarize_cv_metrics(test_metrics_list, save_csv="./results/gmprot_cv_results.csv")

    # fused_embed_extraction()
    # data_util.load_fusion_embeddings(embedding_path="./data/gmprot_ecoli_fusion_embeddings.npy")