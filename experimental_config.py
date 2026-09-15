import json
from dataclasses import dataclass, asdict, field

@dataclass
class ExperimentConfig:
    PROTT5_EMB_DIM = 1024
    
    ecoli_prott5_embedding_file="./data/ecoli_prott5.npz"
    ecoli_blosum_csv="./data/ecoli_blosum62_features.csv"
    ecoli_physio_csv="./data/ecoli_phyiochem.csv"
    ecoli_sinusoidal_csv="./data/ecoli_sinusoidal_encoding.csv"
    ecoli_dataset_csv= "./data/ecoli_five_fold_dataset.csv"#"./data/ecoli_clustered_dataset_splits.csv"

    # Training
    batch_size: int = 1024 #256
    epochs: int = 500
    lr: float = 3e-4
    seed: int = 42 #6, 0 , 3 , 7, 20, 35, 29, 27, 17, 42
    seeds: list = field(
        default_factory=lambda: [6, 0 , 3 , 7, 20, 35, 29, 27, 17, 42]
    )
    patience: int = 300

    #physio features
    physio_feature_dim: int = 32 
    blosum_feature_dim: int = 20

    #Transformer 
    transformer_heads: int = 8
    transformer_ff_dim : int = 512

    # Sequence branch
    seq_dense1: int = 512
    seq_dense2: int = 256
    seq_bottleneck_dim: int = 128
    seq_dropout: float = 0.3

    # GNN
    gnn_base_output_dim: int = 1 #
    gnn_hidden: int = 64
    gnn_heads: int = 4
    gnn_layers: int = 3
    gnn_dense: int = 128
    gnn_bottleneck_dim : int = 128

    # Physio
    physio_proj: int = 32
    pos_proj: int = 12
    physio_bottleneck_dim: int = 128

    # Fusion
    fuse_dense: int = 256
    fuse_dropout: float = 0.3

    #shuffle buffer size
    shuffle_buffer_size: int = 5000

    #huber pearson loss delta
    huber_delta: float = 1.0
    alpha: float = 0.15
    use_rank_loss: bool = False
    beta: float = 0.05 #weight for soft rank loss
    epsilon: float = 1e-6 #numerical stability in pearson calculation(smoothing)

    #Attention Visualization graphs
    visualization_path : str = "./visualization/atten"

    #*************Activation Functions Dual Attention Fusion Train.py***********# 
    #Sequence branch Activation
    seq_proj_A: str = "gelu"
    seq_out_A: str = "gelu"
    seq_bottleneck_A: str = "gelu"
    seq_carp_bottleneck_A: str = "gelu"

    #Graph Branch Activation
    gnn_proj_A: str = "gelu"
    gnn_bottleneck_A: str = "gelu"

    #Physio Branch Activation
    physio_proj_A: str = "gelu"
    physio_bottleneck_A: str = "gelu"


    #**************Model_Dual.py activations****************#
    #Transformer Encoder Readout activations
    ffn_A: str = "gelu"
    
    #GraphAttentionNetwork
    x_A: str = "relu"

    #Position encoding extractor
    sinusoidal_feature_dim : int = 32 #Sinusoidal position encoding dimension
    seq_max_len : int = 128  #Maximum sequence length for positional encoding
    postion_aware_feature_dim : int = 60 #Position aware one-hot encoding dimension

    #cnn 
    cnn_input_dim : int = 21
    cnn_emb_output_dim : int = 24
    cnn_filters : int = 64
    cnn_gating_threshold : float = 0.20
    
    
    EMB_DIM_PROTT5 = 1024




    def save_config(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)