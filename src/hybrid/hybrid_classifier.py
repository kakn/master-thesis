# hybrid_classifier.py

import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils import load_balanced_dataset, print_metrics


class HybridClassifier:
    def __init__(self, use_fine_tuned: bool = False, pooling: str = "first_token", subset_size=None):
        """
        Args:
            use_fine_tuned (bool): Whether to load fine-tuned embeddings or pretrained.
            pooling (str): "first_token" (CLS alternative), "mean_pool" (average of all tokens),
                          or "max_pool" (maximum value across tokens for each dimension).
        """
        self.use_fine_tuned = use_fine_tuned
        self.pooling = pooling
        self.subset_size = subset_size

        self.model_save_path = 'data/saved_models/hybrid_torch'
        self._ensure_directory_exists(self.model_save_path)

        self.model_file = self._generate_model_file_name()
        print(f"Model will be saved to/loaded from: {self.model_file}")

        # Updated architecture variables for the simpler model
        self.hidden_dims = [512, 256, 128]   # New architecture: list of hidden dimensions
        self.dropout = 0.3                   # Dropout probability
        self.hidden_dim = 256  # Hidden layer dimension for the simpler model
        self.output_dim = 1    # Output layer dimension for binary classification
        self.batch_size = 32
        self.lr = 1e-3

        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

        if use_fine_tuned:
            self.embedding_path = "data/llm/finetuned/uncompressed/full_dataset_finetuned_embeddings"
        else:
            self.embedding_path = "data/llm/pretrained/uncompressed/full_dataset_pretrained_embeddings"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _generate_model_file_name(self) -> str:
        """
        Generates a unique model file name based on the model's attributes.
        """
        fine_tuned_tag = "fine_tuned" if self.use_fine_tuned else "pretrained"
        subset_tag = f"subset_{self.subset_size}" if self.subset_size else "full"
        model_file_name = f"hybrid_model_{fine_tuned_tag}_{self.pooling}_{subset_tag}.pt"
        return os.path.join(self.model_save_path, model_file_name)

    def _ensure_directory_exists(self, path):
        os.makedirs(path, exist_ok=True)

    def load_embeddings(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding path not found: {path}")

        hidden_path = os.path.join(path, "hidden_states.npy")
        ids_path = os.path.join(path, "ids.npy")
        if not os.path.exists(hidden_path) or not os.path.exists(ids_path):
            raise FileNotFoundError(f"Expected hidden_states.npy and ids.npy in directory: {path}")

        print(f"Loading embeddings from directory: {path}")
        hidden_states = np.load(hidden_path, mmap_mode="r")
        ids = np.load(ids_path, allow_pickle=True)

        total_samples = hidden_states.shape[0]
        if self.subset_size and self.subset_size < total_samples: 
            total_samples = self.subset_size

        chunk_size = 10000
        all_pooled = []
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk = hidden_states[start_idx:end_idx]

            if self.pooling == "first_token":
                pooled_chunk = chunk[:, 0, :]
            elif self.pooling == "mean_pool":
                pooled_chunk = chunk.mean(axis=1)
            elif self.pooling == "max_pool":
                pooled_chunk = chunk.max(axis=1)
            else:
                raise ValueError("Pooling must be 'first_token', 'mean_pool', or 'max_pool'")

            all_pooled.append(pooled_chunk)

        X_emb_all = np.concatenate(all_pooled, axis=0)
        ids_all = ids[:total_samples]

        y_all = self.get_labels_from_ids(ids_all)

        print(f"Embeddings loaded with shape {X_emb_all.shape}")
        return X_emb_all, y_all

    def get_labels_from_ids(self, ids: np.ndarray) -> np.ndarray:
        """Given an array of IDs, fetch the corresponding labels."""
        dataset = load_balanced_dataset()
        dataset_df = dataset.to_pandas()
        dataset_df.rename(columns={"label": "ai_generated"}, inplace=True)
        dataset_df = dataset_df[["id", "text", "ai_generated"]]
        label_lookup = dataset_df.set_index("id")["ai_generated"].to_dict()
        labels = np.array([label_lookup[i] for i in ids])
        return labels

    def load_handcrafted_features(self) -> Tuple[np.ndarray, np.ndarray]:
        output_dir = "data/saved_data"
        subset_tag = "full"
        features_save_path = os.path.join(output_dir, subset_tag, "features.pkl")
        labels_save_path = os.path.join(output_dir, subset_tag, "labels.pkl")

        if not os.path.exists(features_save_path) or not os.path.exists(labels_save_path):
            raise FileNotFoundError(f"Missing features or labels in {output_dir}/{subset_tag}. Run feature extraction first.")

        print(f"Loading handcrafted features from {features_save_path}...")
        with open(features_save_path, 'rb') as f:
            X = pickle.load(f)

        print(f"Loading labels from {labels_save_path}...")
        with open(labels_save_path, 'rb') as f:
            y = pickle.load(f)

        if self.subset_size and self.subset_size < X.shape[0]:
            print(f"⚡ Using subset of size {self.subset_size} out of {X.shape[0]}")
            X = X[:self.subset_size]
            y = y[:self.subset_size]

        print(f"Handcrafted features loaded with shape {X.shape}")
        return X, y

    def concatenate_features(self, X: np.ndarray, X_emb: np.ndarray) -> np.ndarray:
        """Concatenates handcrafted features with embeddings."""
        if X.shape[0] != X_emb.shape[0]:
            raise ValueError(f"Shape mismatch: X {X.shape}, X_emb {X_emb.shape}")

        merged_X = np.concatenate((X, X_emb), axis=1)
        print(f"Features concatenated. New shape: {merged_X.shape}")
        return merged_X

    def check_label_match(self, y1: np.ndarray, y2: np.ndarray, split_name: str) -> None:
        """Checks if the labels match exactly. Prints debug info if not."""
        mismatches = np.where(y1 != y2)[0]
        if mismatches.size > 0:
            print(f"Warning: {mismatches.size} mismatches found in {split_name} labels!")
            # Print first 20 mismatch indices and their values
            n_to_print = 20
            print(f"Showing first {n_to_print} mismatches:")
            for i in mismatches[:n_to_print]:
                print(f"Index {i}: handcrafted label = {y1[i]}, embedding label = {y2[i]}")
            # Optionally, print statistics on mismatches
            unique_hc, counts_hc = np.unique(y1[mismatches], return_counts=True)
            unique_emb, counts_emb = np.unique(y2[mismatches], return_counts=True)
            print("Mismatch label distribution in handcrafted labels:")
            print(dict(zip(unique_hc, counts_hc)))
            print("Mismatch label distribution in embedding labels:")
            print(dict(zip(unique_emb, counts_emb)))
            raise ValueError(f"Warning: {mismatches.size} mismatches found in {split_name} labels!")
        else:
            print(f"Labels match perfectly for {split_name} split.")

    def load_data(self) -> None:
        X_hc, y_hc = self.load_handcrafted_features()
        X_emb, y_emb = self.load_embeddings(self.embedding_path)

        if X_hc.shape[0] != X_emb.shape[0]:
            raise ValueError(f"Handcrafted rows ({X_hc.shape[0]}) != Embedding rows ({X_emb.shape[0]}). "
                            "They must match one-to-one.")

        if not np.array_equal(y_hc, y_emb):
            mismatch_count = np.sum(y_hc != y_emb)
            raise ValueError(f"{mismatch_count} label mismatches found between handcrafted and embeddings. "
                            "Make sure everything is aligned in ascending ID order.")

        X_full = np.concatenate([X_hc, X_emb], axis=1)
        y_full = y_hc
        
        X_train, X_temp, y_train, y_temp = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        print(f"Final shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    def load_model(self):
        if os.path.exists(self.model_file):
            print(f"Loading model from {self.model_file}...")
            if self.X_train is None:
                self.load_data()
            if self.model is None:
                self.build_simple_model()  # Use the simpler model by default

            # Load weights onto the correct device
            self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
            self.model.to(self.device)
            return True
        return False

    def save_model(self):
        print(f"Saving model to {self.model_file}...")
        torch.save(self.model.state_dict(), self.model_file)  # Save only the state_dict for the PyTorch model

    def build_model(self) -> None:
        """
        Initializes the deeper feed-forward neural network with multiple hidden layers,
        batch normalization, ReLU activations, and dropout.
        """
        input_dim = self.X_train.shape[1]
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.model = nn.Sequential(*layers).to(self.device)
        print(f"Model built with architecture:\n{self.model}")

    def build_simple_model(self) -> None:
        """
        Initializes a simpler feed-forward neural network with a single hidden layer.
        """
        input_dim = self.X_train.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        ).to(self.device)
        print(f"Simple model built with architecture:\n{self.model}")

    def train(self, batch_size=32, lr=1e-3, patience=10):
        """
        Trains the neural network with early stopping and no fixed epoch count.

        Args:
            batch_size (int): Batch size.
            lr (float): Learning rate.
            patience (int): Number of epochs to wait for improvement before stopping early.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call `build_model()` or `build_simple_model()` first.")

        train_loader = DataLoader(
            TensorDataset(torch.tensor(self.X_train, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(self.X_val, dtype=torch.float32), torch.tensor(self.y_val, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_loss, patience_counter = float('inf'), 0
        best_model_state = None
        epoch = 0

        while True:  # Runs indefinitely until early stopping
            epoch += 1
            self.model.train()
            total_train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            
            for X, y in progress_bar:
                X, y = X.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X).squeeze(1)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation Step
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    logits = self.model(X).squeeze(1)
                    loss = criterion(logits, y)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early Stopping
            if avg_val_loss < best_loss:
                best_loss, patience_counter, best_model_state = avg_val_loss, 0, self.model.state_dict()
                print("Validation loss improved! Saving model...")
            else:
                patience_counter += 1
                print(f"No improvement. Early stopping counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered. Restoring best model.")
                self.model.load_state_dict(best_model_state)
                break  # Exits training loop

    def evaluate(self) -> None:
        """
        Evaluates the trained model on the test set and prints additional metrics.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call `build_model()` or `build_simple_model()` first.")

        test_dataset = TensorDataset(torch.tensor(self.X_test, dtype=torch.float32), 
                                     torch.tensor(self.y_test, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        self.model.eval()

        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
            
            for X_batch, y_batch in progress_bar:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                logits = self.model(X_batch).squeeze(1)
                loss = criterion(logits, y_batch)
                total_loss += loss.item()

                probs = torch.sigmoid(logits)  # Convert logits to probabilities
                preds = (probs >= 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}")
        print_metrics(all_labels, all_preds, verbose=True)
    
    def evaluate_evasive_texts(self) -> None:
        """
        Evaluates the trained model on evasive texts (control, basic, advanced).
        """
        if self.model is None:
            raise ValueError("Model is not built. Call `build_model()` or `build_simple_model()` first.")

        # Load human texts from the test set
        human_dataset = TensorDataset(torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.float32))
        human_loader = DataLoader(human_dataset, batch_size=32, shuffle=False)

        # Extract human texts and labels
        human_X, human_y = [], []
        with torch.no_grad():
            for X_batch, y_batch in human_loader:
                human_X.extend(X_batch.cpu().numpy())
                human_y.extend(y_batch.cpu().numpy())

        human_X = np.array(human_X)
        human_y = np.array(human_y)

        evasive_types = ["control", "basic", "advanced"]

        for e_t in evasive_types:
            print(f"\nEvaluating on {e_t.capitalize()} evasive texts...")

            # Load embeddings and their IDs
            embedding_path = f"data/llm/pretrained/uncompressed/{e_t}_pretrained_embeddings"
            X_emb_evasive, ids_evasive = self.load_embeddings(embedding_path)

            # Load handcrafted features and their IDs
            feature_path = f"data/evasive_texts/feature/{e_t}/full_features.pkl"
            ids_hc_path = f"data/evasive_texts/feature/{e_t}/full_ids.pkl"
            labels_path = f"data/evasive_texts/feature/{e_t}/full_labels.pkl"

            if not all(os.path.exists(p) for p in [feature_path, ids_hc_path, labels_path]):
                raise FileNotFoundError(f"Missing files for {e_t} evasive texts.")

            with open(feature_path, 'rb') as f:
                X_evasive_hc = pickle.load(f)
            with open(ids_hc_path, 'rb') as f:
                ids_hc = pickle.load(f)
            with open(labels_path, 'rb') as f:
                y_evasive = pickle.load(f)

            # Verify ID alignment instead of label check
            if not np.array_equal(ids_evasive, ids_hc):
                mismatch_idx = np.where(ids_evasive != ids_hc)[0]
                raise ValueError(
                    f"ID mismatch between embeddings and features for {e_t} evasive texts. "
                    f"First mismatch at index {mismatch_idx[0]}: "
                    f"Embedding ID={ids_evasive[mismatch_idx[0]]}, "
                    f"Feature ID={ids_hc[mismatch_idx[0]]}"
                )

            # Merge: handcrafted + embeddings
            X_evasive_combined = np.concatenate((X_evasive_hc, X_emb_evasive), axis=1)

            # Truncate human texts to match length of evasive
            human_X_truncated = human_X[:len(X_evasive_combined)]
            human_y_truncated = human_y[:len(X_evasive_combined)]

            # Combine them
            combined_X = np.concatenate((human_X_truncated, X_evasive_combined), axis=0)
            combined_y = np.concatenate((human_y_truncated, y_evasive), axis=0)

            # Evaluate
            combined_dataset = TensorDataset(torch.tensor(combined_X, dtype=torch.float32), torch.tensor(combined_y, dtype=torch.float32))
            combined_loader = DataLoader(combined_dataset, batch_size=32, shuffle=False)

            self.model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in combined_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    logits = self.model(X_batch).squeeze(1)
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).long().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y_batch.cpu().numpy())

            print(f"Evaluation for {e_t.capitalize()} evasive texts:")
            print_metrics(all_labels, all_preds, verbose=True)

    def run(self, force_retrain: bool = False, use_simple_model: bool = True) -> None:
        """Runs the full pipeline: data loading, model initialization, training, and evaluation."""
        print("Running full hybrid classifier pipeline...")
        if self.load_model() and not force_retrain:
            print("Model loaded successfully. Skipping training.")
        else:
            self.load_data()
            if use_simple_model:
                self.build_simple_model()
            else:
                self.build_model()
            self.train(batch_size=self.batch_size, lr=self.lr)
            self.save_model()
        self.evaluate()
        self.evaluate_evasive_texts()

def train_hybrid_model(subset_size = None, use_simple_model: bool = True, use_fine_tuned: bool = False, pooling: str = "first_token") -> None:
    """Initializes and runs the HybridClassifier."""
    print("Initializing hybrid classifier...")
    hybrid_model = HybridClassifier(subset_size=subset_size, use_fine_tuned=use_fine_tuned, pooling=pooling)
    hybrid_model.run(use_simple_model=use_simple_model)

def debug_none_ids_thorough():
    """Identifies the exact source of None IDs in embedding files."""
    # 1. Check original dataset's ID validity
    print("🔍 Checking original dataset...")
    dataset = load_balanced_dataset()
    original_ids = [ex['id'] for ex in dataset]
    num_none_original = sum(1 for _id in original_ids if _id is None)
    print(f"Original dataset has {num_none_original} None IDs\n")

    # 2. Check main embedding files
    print("🔍 Checking main dataset embeddings...")
    main_emb_dir = "data/llm/pretrained/uncompressed/full_dataset_pretrained_embeddings"
    main_ids = np.load(os.path.join(main_emb_dir, "ids.npy"), allow_pickle=True)
    main_none_mask = main_ids == None
    print(f"Main embeddings contain {np.sum(main_none_mask)} None IDs")

    if np.any(main_none_mask):
        print("\n🚨 Main embeddings have None IDs! Investigating origin...")
        # Find first 3 problematic indices
        bad_indices = np.where(main_none_mask)[0][:3]
        for idx in bad_indices:
            print(f"\nIndex {idx}:")
            # Reconstruct original dataset position
            original_idx = idx  # Only valid if subset_size=None and no shuffling
            print(f"Original dataset ID: {original_ids[original_idx]}")
            print(f"Original text snippet: {dataset[original_idx]['text'][:50]}...")

    # 3. Check evasive datasets
    print("\n🔍 Checking evasive CSVs...")
    for name in ["control", "basic", "advanced"]:
        csv_path = f"data/evasive_texts/{name}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            num_none = df['id'].isnull().sum()
            print(f"{name} CSV: {num_none} None IDs")
            if num_none > 0:
                print(f"First bad row:\n{df[df['id'].isnull()].iloc[0]}")
        else:
            print(f"{name} CSV not found")

if __name__ == "__main__":
    train_hybrid_model()