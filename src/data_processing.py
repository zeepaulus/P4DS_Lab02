import numpy as np

def load_data(filepath):
    raw_rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        header = lines[0].strip().replace('"', '').split(',')
        for line in lines[1:]:
            row = line.strip().replace('"', '').split(',')
            raw_rows.append(row)
            
    raw_data_str = np.array(raw_rows)
    n_rows, n_cols = raw_data_str.shape
    data = np.empty((n_rows, n_cols), dtype=object)

    for i in range(n_cols):
        col_vals = raw_data_str[:, i]
        try:
            float_vals = col_vals.astype(float)
            is_integer = np.all(np.mod(float_vals, 1) == 0) and not np.isnan(float_vals).any()
            if is_integer:
                data[:, i] = float_vals.astype(int)
            else:
                data[:, i] = float_vals
        except ValueError:
            data[:, i] = col_vals
    return data, header

def get_column_index(header, col_name):
    return header.index(col_name)

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize(X):
    X = X.astype(float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def one_hot_encode(column_data):
    unique_vals = np.unique(column_data)
    n_samples = len(column_data)
    n_classes = len(unique_vals)
    one_hot = np.zeros((n_samples, n_classes), dtype=int)
    for i, val in enumerate(unique_vals):
        one_hot[:, i] = (column_data == val).astype(int)
    return one_hot, unique_vals