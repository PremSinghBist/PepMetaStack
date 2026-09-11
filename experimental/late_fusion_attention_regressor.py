import copy
import pickle
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Custom project imports
from recomb_code.data_retriever import DataRetriever


# =====================================================================
# 1. PyTorch Neural Network Module: Attention-Pooling Architecture
# =====================================================================
class LateFusionAttentionModel(nn.Module):
    """
    Applies Attention Pooling over 2D residue embeddings (L x D_emb),
    then concatenates static 1D global features before the MLP regressor.
    """
    def __init__(self, emb_dim=1024, global_dim=84, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # 1. Attention Scoring Mechanism over sequence length (L)
        self.attention_net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 2. Combined Dimension after Late Fusion
        total_input_dim = emb_dim + global_dim
        
        # 3. Deep MLP Head
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.ln2 = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(dropout)
        
        self.out = nn.Linear(64, 1)

    def forward(self, x_emb, mask, x_global):
        """
        x_emb    : Tensor (Batch, L, emb_dim)
        mask     : BoolTensor (Batch, L)
        x_global : Tensor (Batch, global_dim)
        """
        # Calculate raw scalar attention scores per residue position
        attn_logits = self.attention_net(x_emb).squeeze(-1)  # (Batch, L)
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, -1e9)
            
        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (Batch, L, 1)
        
        # Weighted sum pooling across sequence positions L -> (Batch, emb_dim)
        pooled_emb = torch.sum(x_emb * attn_weights, dim=1)
        
        # Late Fusion: Concatenate pooled embeddings with static global features
        if x_global.size(1) > 0:
            fusion_vector = torch.cat([pooled_emb, x_global], dim=1)
        else:
            fusion_vector = pooled_emb
        
        # Deep MLP Regression
        h = F.silu(self.ln1(self.fc1(fusion_vector)))
        h = self.drop1(h)
        
        h = F.silu(self.ln2(self.fc2(h)))
        h = self.drop2(h)
        
        prediction = self.out(h).squeeze(-1)
        return prediction, attn_weights


# =====================================================================
# 2. PyTorch Dataset & Collate Function
# =====================================================================
class MultiModalDataset(Dataset):
    def __init__(self, emb_list, global_list, y_targets):
        self.emb_list = emb_list
        self.global_list = global_list
        self.y_targets = y_targets

    def __len__(self):
        return len(self.emb_list)

    def __getitem__(self, idx):
        return self.emb_list[idx], self.global_list[idx], self.y_targets[idx]


def collate_multimodal_batch(batch):
    emb_list, global_list, y_list = zip(*batch)
    
    lengths = [e.shape[0] for e in emb_list]
    max_len = max(lengths)
    emb_dim = emb_list[0].shape[1]
    batch_size = len(emb_list)
    
    padded_emb = torch.zeros((batch_size, max_len, emb_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    for i, e in enumerate(emb_list):
        seq_len = e.shape[0]
        padded_emb[i, :seq_len, :] = torch.from_numpy(e)
        mask[i, :seq_len] = True
        
    global_tensor = torch.tensor(np.array(global_list), dtype=torch.float32)
    targets = torch.tensor(y_list, dtype=torch.float32)
    
    return padded_emb, mask, global_tensor, targets


# =====================================================================
# 3. Pipeline Class
# =====================================================================
class AttentionPoolingRegressionPipeline:
    def __init__(self, epochs=60, lr=1e-3, batch_size=32, feature_flags=None, device=None):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        if feature_flags is None:
            self.feature_flags = {
                "include_emb": True,
                "include_physio": True,
                "include_blosum": True,
                "include_sinu": True,
            }
        else:
            self.feature_flags = feature_flags

    @staticmethod
    def extract_features(record, include_emb=True, include_physio=True, include_blosum=True, include_sinu=True):
        """
        Separates 2D sequence matrix (emb) from 1D static vectors (physio, blosum, sinu).
        Respects all feature flags including `include_emb`.
        """
        emb, physio, blosum, sinu, seq, mic = record
        seq_len = len(seq)
        
        # 1. Sequence-level matrix (L, 1024)
        if include_emb:
            emb_matrix = np.asarray(emb, dtype=np.float32)
            if emb_matrix.ndim == 1:
                emb_matrix = np.expand_dims(emb_matrix, axis=0)

            # Truncate/align embedding sequence length L to sequence length string if off-by-one
            if emb_matrix.shape[0] > seq_len:
                emb_matrix = emb_matrix[:seq_len, :]
        else:
            # If sequence embedding is toggled off, return zero matrix placeholder of shape (seq_len, 1)
            emb_matrix = np.zeros((seq_len, 1), dtype=np.float32)

        # 2. Global static vectors (concat into 1D array)
        global_feats = []
        if include_physio:
            global_feats.append(np.asarray(physio, dtype=np.float32).ravel())
        if include_blosum:
            global_feats.append(np.asarray(blosum, dtype=np.float32).ravel())
        if include_sinu:
            global_feats.append(np.asarray(sinu, dtype=np.float32).ravel())

        if global_feats:
            global_vector = np.concatenate(global_feats)
        else:
            global_vector = np.empty((0,), dtype=np.float32)

        return emb_matrix, global_vector, float(mic)

    @classmethod
    def prepare_split(cls, dataset_split, **feature_flags):
        emb_list, global_list, y_list = [], [], []
        for record in dataset_split:
            emb_mat, glob_vec, y_val = cls.extract_features(record, **feature_flags)
            emb_list.append(emb_mat)
            global_list.append(glob_vec)
            y_list.append(y_val)
            
        return emb_list, global_list, np.array(y_list, dtype=np.float32)

    @staticmethod
    def scale_features(train_tuple, val_tuple, test_tuple):
        """Fits StandardScaler separately on embeddings and global vectors."""
        tr_emb, tr_glob, tr_y = train_tuple
        val_emb, val_glob, val_y = val_tuple
        te_emb, te_glob, te_y = test_tuple

        # Scale Embeddings (Residue level)
        scaler_emb = StandardScaler()
        scaler_emb.fit(np.vstack(tr_emb))
        
        tr_emb_s = [scaler_emb.transform(x) for x in tr_emb]
        val_emb_s = [scaler_emb.transform(x) for x in val_emb]
        te_emb_s = [scaler_emb.transform(x) for x in te_emb]

        # Scale Global Vectors
        if tr_glob[0].size > 0:
            scaler_glob = StandardScaler()
            scaler_glob.fit(np.array(tr_glob))
            
            tr_glob_s = scaler_glob.transform(np.array(tr_glob))
            val_glob_s = scaler_glob.transform(np.array(val_glob))
            te_glob_s = scaler_glob.transform(np.array(te_glob))
        else:
            tr_glob_s, val_glob_s, te_glob_s = tr_glob, val_glob, te_glob

        return (tr_emb_s, tr_glob_s, tr_y), (val_emb_s, val_glob_s, val_y), (te_emb_s, te_glob_s, te_y)

    def run_attention_pipeline(self, data_retriever, metric_evaluator=None):
        folds = data_retriever.load_dataset_pickle()
        print(f"Loaded {len(folds)} folds. Execution Device: {str(self.device).upper()}\n")

        test_metrics_list = []

        for fold_idx, (train_set, val_set, test_set) in enumerate(folds):
            print(f"--- Training Fold {fold_idx + 1}/{len(folds)} ---")

            tr_data = self.prepare_split(train_set, **self.feature_flags)
            val_data = self.prepare_split(val_set, **self.feature_flags)
            te_data = self.prepare_split(test_set, **self.feature_flags)

            (tr_emb, tr_glob, y_tr), (val_emb, val_glob, y_val), (te_emb, te_glob, y_te) = \
                self.scale_features(tr_data, val_data, te_data)

            emb_dim = tr_emb[0].shape[1]
            global_dim = tr_glob[0].shape[0] if tr_glob[0].size > 0 else 0

            train_loader = DataLoader(
                MultiModalDataset(tr_emb, tr_glob, y_tr),
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_multimodal_batch
            )
            val_loader = DataLoader(
                MultiModalDataset(val_emb, val_glob, y_val),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_multimodal_batch
            )
            test_loader = DataLoader(
                MultiModalDataset(te_emb, te_glob, y_te),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_multimodal_batch
            )

            model = LateFusionAttentionModel(emb_dim=emb_dim, global_dim=global_dim).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-2)
            criterion = nn.SmoothL1Loss()

            best_val_loss = float("inf")
            best_model_state = copy.deepcopy(model.state_dict())

            for epoch in range(self.epochs):
                model.train()
                train_loss, train_samples = 0.0, 0
                
                # Wrap train_loader; leave=False removes the batch bar after epoch ends
                batch_pbar = tqdm(
                    train_loader, 
                    desc=f"Epoch {epoch + 1:02d}/{self.epochs:02d}", 
                    leave=False
                ) # type: ignore
                
                for x_e, mask, x_g, y in batch_pbar:
                    x_e, mask, x_g, y = x_e.to(self.device), mask.to(self.device), x_g.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    preds, _ = model(x_e, mask, x_g)
                    loss = criterion(preds, y)
                    loss.backward()
                    optimizer.step()
                    
                    b_size = x_e.size(0)
                    train_loss += loss.item() * b_size
                    train_samples += b_size
                    
                    # Display immediate loss for the current batch
                    batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

                train_loss /= train_samples

                # --- Validation Loop ---
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for x_e, mask, x_g, y in val_loader:
                        x_e, mask, x_g, y = x_e.to(self.device), mask.to(self.device), x_g.to(self.device), y.to(self.device)
                        preds, _ = model(x_e, mask, x_g)
                        
                        batch_size = x_e.size(0)
                        val_loss += criterion(preds, y).item() * batch_size

                val_loss /= len(val_loader.dataset) #type: ignore

                # Checkpoint the best model parameters
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Fold {fold_idx + 1} | Epoch {epoch + 1:02d}/{self.epochs:02d} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val: {best_val_loss:.4f}")

            # --- Final Test Evaluation using Best Checkpoint ---
            model.load_state_dict(best_model_state)
            model.eval()
            te_preds = []
            with torch.no_grad():
                for x_e, mask, x_g, _ in test_loader:
                    x_e, mask, x_g = x_e.to(self.device), mask.to(self.device), x_g.to(self.device)
                    preds, _ = model(x_e, mask, x_g)
                    te_preds.extend(preds.cpu().numpy())
                    
            best_test_preds = np.array(te_preds)

            if metric_evaluator:
                test_m = metric_evaluator.compute_metrics(y_te, best_test_preds)
                test_metrics_list.append(test_m)
                print(f"Fold {fold_idx + 1} Test Results -> Pearson R: {test_m['Pearson_R']:.4f} | RMSE: {test_m['RMSE']:.4f}")

        if test_metrics_list and metric_evaluator:
            print("\n" + "=" * 65)
            print("=== FINAL SUMMARY (LATE FUSION ATTENTION) ===")
            print("=" * 65)
            for k in ["Pearson_R", "Spearman_R", "RMSE", "R2"]:
                vals = [m[k] for m in test_metrics_list]
                print(f"  ├── {k:<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


# =====================================================================
# Metric Evaluator Class
# =====================================================================
class MetricEvaluator:
    """Computes regression metrics: Pearson R, Spearman R, Kendall Tau, RMSE, MAE, R2."""
    @staticmethod
    def compute_metrics(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        spearman_r, _ = stats.spearmanr(y_true, y_pred)
        kendall_tau, _ = stats.kendalltau(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            "Pearson_R": pearson_r,
            "Spearman_R": spearman_r,
            "Kendall_Tau": kendall_tau,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }


# =====================================================================
# Main Execution Entry Point
# =====================================================================
if __name__ == "__main__":
    # 1. Set random seeds for reproducible runs
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    DATASET_PATH = "./ecoli_prott5_based_datasets.pkl"

    # 2. Initialize Data Retriever and inspect dataset structure
    data_retriever = DataRetriever(DATASET_PATH)
    data_retriever.explain_data_structure()

    # 3. Define feature flags (select active modalities)
    feature_flags = {
        "include_emb": True,
        "include_physio": True,
        "include_blosum": True,
        "include_sinu": True,
    }

    # 4. Instantiate Metric Evaluator
    metric_evaluator = MetricEvaluator()

    # 5. Instantiate Attention Pooling Pipeline
    pipeline = AttentionPoolingRegressionPipeline(
        epochs=500,
        lr=1e-3,
        batch_size=32,
        feature_flags=feature_flags
    )

    # 6. Execute 5-Fold Training and Evaluation Loop
    pipeline.run_attention_pipeline(
        data_retriever=data_retriever,
        metric_evaluator=metric_evaluator
    )