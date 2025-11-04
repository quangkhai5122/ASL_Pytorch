import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import pandas as pd
import os

try:
    from scripts.config import NUM_CLASSES, BATCH_ALL_SIGNS_N, INPUT_SIZE, N_COLS, N_DIMS
    from scripts.preprocess import PreprocessLayer
    from scripts.utils import load_relevant_data_subset
except ImportError:
    from config import NUM_CLASSES, BATCH_ALL_SIGNS_N, INPUT_SIZE, N_COLS, N_DIMS
    from preprocess import PreprocessLayer
    from utils import load_relevant_data_subset

class ASLParquetDataset(Dataset):
    """
    Dataset loads raw .parquet files and applies preprocessing on the fly.
    """
    def __init__(self, csv_path='data/train.csv', data_root='data/'):
        
        # Handle paths dynamically
        self.csv_path = self._resolve_path(csv_path)
        self.data_root = self._resolve_path(data_root)

        if not os.path.exists(self.csv_path):
            print(f"Error: train.csv not found at {self.csv_path}")
            self.df = pd.DataFrame()
            self.labels = np.array([])
            return

        self.df = pd.read_csv(self.csv_path)
        
        # Ensure 'sign_ord' column exists (needed for the sampler)
        if 'sign_ord' not in self.df.columns:
            self.df['sign_ord'] = self.df['sign'].astype('category').cat.codes
        
        self.labels = self.df['sign_ord'].values
        
        # Initialize the preprocessor. We keep it on CPU as it will be used by DataLoader workers.
        self.preprocess = PreprocessLayer()
        self.preprocess.eval() # Preprocessing is deterministic

    def _resolve_path(self, path):
        # Helper to find the path relative to the project root if the absolute path doesn't exist
        if os.path.exists(path):
            return path
        # Assume script might be run from project root or src/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resolved_path = os.path.join(base_dir, path)
        if os.path.exists(resolved_path):
            return resolved_path
        return path # Return original path if resolution fails

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct the full path to the parquet file
        parquet_path = os.path.join(self.data_root, row['path'])
        label = self.labels[idx]

        # 1. Load raw data (Numpy array)
        raw_data_np = load_relevant_data_subset(parquet_path)
        
        # Handle missing or empty files gracefully
        if raw_data_np.size == 0:
            # Return zero tensors matching the expected output shape of preprocessing
            frames = torch.zeros((INPUT_SIZE, N_COLS, N_DIMS), dtype=torch.float32)
            non_empty_idxs = torch.full((INPUT_SIZE,), -1.0, dtype=torch.float32)
            return frames, non_empty_idxs, torch.tensor(label, dtype=torch.long)

        # 2. Convert to Torch Tensor (on CPU)
        raw_data_torch = torch.tensor(raw_data_np, dtype=torch.float32)

        # 3. Apply Preprocessing
        # Use torch.no_grad() as this is input transformation
        with torch.no_grad():
             frames, non_empty_idxs = self.preprocess(raw_data_torch)

        return frames, non_empty_idxs, torch.tensor(label, dtype=torch.long)

# AllSignsBatchSampler remains the same, as it operates on the labels (y)
class AllSignsBatchSampler(Sampler):
    """
    Custom BatchSampler to replicate the logic of the get_train_batch_all_signs generator.
    """
    def __init__(self, labels, n_samples_per_class=BATCH_ALL_SIGNS_N, num_classes=NUM_CLASSES):
        self.labels = labels
        self.n_samples_per_class = n_samples_per_class
        self.num_classes = num_classes
        self.batch_size = self.num_classes * self.n_samples_per_class
        
        if len(labels) == 0:
            self.class2idxs = {}
            return

        # Create a mapping from class ID to indices in the dataset
        self.class2idxs = {}
        for i in range(self.num_classes):
             # Ensure indices are available for the class
            indices = np.where(self.labels == i)[0]
            if indices.size > 0:
                self.class2idxs[i] = indices

    def __iter__(self):
        if len(self.labels) == 0:
            return iter([])

        # Define an epoch length.
        num_batches = len(self.labels) // self.batch_size
        if num_batches == 0 and len(self.labels) > 0:
            num_batches = 1
        
        # Generate batches for the epoch
        for _ in range(num_batches):
            batch_indices = []
            for i in range(self.num_classes):
                if i in self.class2idxs:
                    # Determine replacement strategy: if not enough samples, allow replacement
                    replace = len(self.class2idxs[i]) < self.n_samples_per_class
                    
                    # Sample indices (mimicking np.random.choice behavior)
                    idxs = np.random.choice(self.class2idxs[i], self.n_samples_per_class, replace=replace)
                    batch_indices.extend(idxs)

            yield batch_indices

    def __len__(self):
        if self.batch_size == 0:
            return 0
        return len(self.labels) // self.batch_size