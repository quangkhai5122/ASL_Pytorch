import pandas as pd
import numpy as np
import os

try:
    from scripts.config import N_ROWS
except ImportError:
    from config import N_ROWS

def load_relevant_data_subset(pq_path):
    """
    Loads the raw parquet data and reshapes it into (n_frames, N_ROWS, 3).
    """
    data_columns = ['x', 'y', 'z']
    try:
        data = pd.read_parquet(pq_path, columns=data_columns)
    except FileNotFoundError:
        print(f"Error: Parquet file not found at {pq_path}")
        return np.array([])

    n_frames = int(len(data) / N_ROWS)
    
    if n_frames == 0:
        return np.array([])
        
    data = data.values.reshape(n_frames, N_ROWS, len(data_columns))
    return data.astype(np.float32)

def load_data_maps(csv_path='data/train.csv'):
    """
    Loads the training CSV and generates sign-to-ordinal mappings.
    """
    # Handle relative path if running from different directories
    if not os.path.exists(csv_path):
         base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
         csv_path = os.path.join(base_path, csv_path)
         # Try again with the absolute path
         if not os.path.exists(csv_path):
            # Final fallback if running app.py from root
            csv_path = 'data/train.csv'
    try:
        train = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return {}, {}

    train['sign_ord'] = train['sign'].astype('category').cat.codes
    
    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
    
    return SIGN2ORD, ORD2SIGN