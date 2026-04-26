"""
Data loading, spatial sorting, overlapping partitions, and scaling.
Covers entire 50k mesh via sliding-window partitions with overlap averaging.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.preprocessing import StandardScaler
from config import Config


def spatial_sort_solution(df_sol):
    """
    Sort mesh points by geodesic spiral on the capsule surface.
    Primary: latitude bands from stagnation point outward.
    Secondary: azimuthal angle within each band.
    """
    x, y, z = df_sol['X'].values, df_sol['Y'].values, df_sol['Z'].values
    r_perp = np.sqrt(y**2 + z**2)
    theta_geo = np.arctan2(r_perp, x)
    phi = np.arctan2(z, y)
    n_bands = 64
    theta_band = np.digitize(theta_geo, np.linspace(0, np.pi, n_bands + 1)) - 1
    sort_idx = np.lexsort((phi, theta_band))
    return sort_idx


def create_partitions(features, targets, seq_len, stride):
    """
    Create overlapping partitions from a spatially-sorted solution.
    Returns list of (features_partition, targets_partition, start_idx, end_idx).
    """
    total = len(features)
    partitions = []
    start = 0
    while start < total:
        end = min(start + seq_len, total)
        f_part = features[start:end]
        t_part = targets[start:end]
        # Pad if partition is shorter than seq_len (last partition)
        if len(f_part) < seq_len:
            pad_len = seq_len - len(f_part)
            f_part = np.concatenate([f_part, np.tile(f_part[-1:], (pad_len, 1))], axis=0)
            t_part = np.concatenate([t_part, np.tile(t_part[-1:], (pad_len, 1))], axis=0)
        partitions.append((f_part, t_part, start, end))
        if end >= total:
            break
        start += stride
    return partitions


def load_and_clean(cfg: Config, file_path: str):
    """Load CSV, assign solution IDs, split, clean."""
    df_raw = pd.read_csv(file_path)
    print(f"Raw dataset shape: {df_raw.shape}")

    n_rows = len(df_raw)
    assert n_rows % cfg.points_per_solution == 0, (
        f"Row count {n_rows} not divisible by {cfg.points_per_solution}"
    )
    n_solutions = n_rows // cfg.points_per_solution
    print(f"Detected solutions: {n_solutions}")

    df_raw = df_raw.copy()
    df_raw["location_id"] = (np.arange(n_rows) // cfg.points_per_solution).astype(np.int32)

    # Random split by solution
    unique_locs = df_raw["location_id"].unique()
    rng = np.random.RandomState(cfg.split_seed)
    rng.shuffle(unique_locs)
    n = len(unique_locs)
    n_train = int(round(cfg.train_frac * n))
    n_val = int(round(cfg.val_frac * n))

    split_map = {}
    for lid in unique_locs[:n_train]:
        split_map[lid] = "train"
    for lid in unique_locs[n_train:n_train + n_val]:
        split_map[lid] = "val"
    for lid in unique_locs[n_train + n_val:]:
        split_map[lid] = "test"
    df_raw["split"] = df_raw["location_id"].map(split_map)

    print("Split counts:")
    print(df_raw["split"].value_counts(dropna=False))

    # Cleaning
    df_clean = df_raw.copy()
    df_clean = df_clean[df_clean['theta (m)'] >= 0]
    df_clean = df_clean[df_clean['Re-theta'] >= 1e-5]
    df_clean = df_clean[(df_clean['qw (W/m^2)'] >= 1e3) & (df_clean['qw (W/m^2)'] <= 1e7)]

    print(f"After cleaning: {len(df_clean):,} rows, "
          f"{df_clean['location_id'].nunique()} solutions")
    return df_clean


def build_partition_dataset(df, split_name, cfg: Config):
    """
    Build all partitions for a split. Each partition is an independent sample.
    Returns (X_partitions, Y_partitions, partition_meta).
    """
    df_split = df[df['split'] == split_name]
    loc_ids = sorted(df_split['location_id'].unique())

    X_parts, Y_parts, metas = [], [], []
    for lid in loc_ids:
        sol = df_split[df_split['location_id'] == lid]
        sort_idx = spatial_sort_solution(sol)
        sol_sorted = sol.iloc[sort_idx]

        feats = sol_sorted[cfg.x_cols].values.astype(np.float32)
        targs = sol_sorted[cfg.y_csv_cols].values.astype(np.float32)

        parts = create_partitions(feats, targs, cfg.seq_len, cfg.partition_stride)
        for f_part, t_part, start, end in parts:
            X_parts.append(f_part)
            Y_parts.append(t_part)
            metas.append({
                'location_id': lid,
                'start': start,
                'end': end,
                'n_points_orig': len(sol),
                'n_valid': min(end, len(sol)) - start,
            })

    return np.stack(X_parts), np.stack(Y_parts), metas


def build_solution_dataset(df, split_name, cfg: Config):
    """
    Build full-solution arrays for evaluation (no partitioning, keep all points).
    Pads/truncates to max_points for batching.
    """
    df_split = df[df['split'] == split_name]
    loc_ids = sorted(df_split['location_id'].unique())

    all_feats, all_targs = [], []
    for lid in loc_ids:
        sol = df_split[df_split['location_id'] == lid]
        sort_idx = spatial_sort_solution(sol)
        sol_sorted = sol.iloc[sort_idx]
        feats = sol_sorted[cfg.x_cols].values.astype(np.float32)
        targs = sol_sorted[cfg.y_csv_cols].values.astype(np.float32)
        all_feats.append(feats)
        all_targs.append(targs)

    return all_feats, all_targs, loc_ids


def fit_scalers(X_train, Y_train, cfg: Config):
    """Fit StandardScalers on training data. Targets are log-transformed first."""
    scaler_X = StandardScaler()
    X_flat = X_train.reshape(-1, cfg.n_features)
    scaler_X.fit(X_flat)

    scaler_y = StandardScaler()
    Y_log = transform_targets_for_training(X_train, Y_train, cfg)
    Y_flat = Y_log.reshape(-1, cfg.n_outputs)
    scaler_y.fit(Y_flat)

    return scaler_X, scaler_y


def transform_targets_for_training(X, Y, cfg: Config):
    """
    Convert physical targets to the log-space quantities optimized by the model.

    By default every target is log10(y). When normalize_qw_by_rhov3 is enabled,
    the qw target becomes log10(qw / (rho * V^3)), matching the pointwise
    physics-normalized heat-flux experiment.
    """
    Y_log = np.log10(np.clip(Y, 1e-6, None)).astype(np.float32)

    if cfg.normalize_qw_by_rhov3 and 'qw' in cfg.y_col_names:
        qw_idx = cfg.y_col_names.index('qw')
        velocity = X[:, :, 3]
        density = X[:, :, 4]
        phys_log = np.log10(
            np.clip(density * np.power(velocity, 3), 1e-12, None)
        ).astype(np.float32)
        Y_log[:, :, qw_idx] = Y_log[:, :, qw_idx] - phys_log

    return Y_log


def apply_scalers(X, Y, scaler_X, scaler_y, cfg: Config):
    """Apply fitted scalers. Returns scaled arrays."""
    n, sl = X.shape[0], X.shape[1]
    X_s = scaler_X.transform(X.reshape(-1, cfg.n_features)).reshape(n, sl, cfg.n_features).astype(np.float32)
    Y_log = transform_targets_for_training(X, Y, cfg)
    Y_s = scaler_y.transform(Y_log.reshape(-1, cfg.n_outputs)).reshape(n, sl, cfg.n_outputs).astype(np.float32)
    return X_s, Y_s


class CFDPartitionDataset(Dataset):
    """Dataset of partition sequences for training."""
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def _load_and_clean_quiet(cfg: Config, file_path: str):
    """Same as load_and_clean but without print statements (for non-main ranks)."""
    df_raw = pd.read_csv(file_path)
    n_rows = len(df_raw)
    assert n_rows % cfg.points_per_solution == 0
    df_raw = df_raw.copy()
    df_raw["location_id"] = (np.arange(n_rows) // cfg.points_per_solution).astype(np.int32)
    unique_locs = df_raw["location_id"].unique()
    rng = np.random.RandomState(cfg.split_seed)
    rng.shuffle(unique_locs)
    n = len(unique_locs)
    n_train = int(round(cfg.train_frac * n))
    n_val = int(round(cfg.val_frac * n))
    split_map = {}
    for lid in unique_locs[:n_train]:
        split_map[lid] = "train"
    for lid in unique_locs[n_train:n_train + n_val]:
        split_map[lid] = "val"
    for lid in unique_locs[n_train + n_val:]:
        split_map[lid] = "test"
    df_raw["split"] = df_raw["location_id"].map(split_map)
    df_clean = df_raw.copy()
    df_clean = df_clean[df_clean['theta (m)'] >= 0]
    df_clean = df_clean[df_clean['Re-theta'] >= 1e-5]
    df_clean = df_clean[(df_clean['qw (W/m^2)'] >= 1e3) & (df_clean['qw (W/m^2)'] <= 1e7)]
    return df_clean


def get_dataloaders(cfg: Config, file_path: str, distributed=False, rank=0, world_size=1):
    """
    Full pipeline: load → clean → partition → scale → dataloaders.
    Returns (train_dl, val_dl, test_dl, scaler_X, scaler_y, test_raw_feats, test_raw_targs).
    """
    verbose = (rank == 0)
    df = load_and_clean(cfg, file_path) if verbose else _load_and_clean_quiet(cfg, file_path)

    # Build partitioned datasets for train/val
    if verbose:
        print("Building partitioned datasets...")
    X_train_raw, Y_train_raw, meta_train = build_partition_dataset(df, 'train', cfg)
    X_val_raw, Y_val_raw, meta_val = build_partition_dataset(df, 'val', cfg)
    X_test_raw, Y_test_raw, meta_test = build_partition_dataset(df, 'test', cfg)

    if verbose:
        print(f"Train partitions: {X_train_raw.shape}")
        print(f"Val partitions:   {X_val_raw.shape}")
        print(f"Test partitions:  {X_test_raw.shape}")
        print(f"Partitions per solution: {cfg.n_partitions}")

    # Fit scalers on training data
    scaler_X, scaler_y = fit_scalers(X_train_raw, Y_train_raw, cfg)

    # Apply scaling
    X_train_s, Y_train_s = apply_scalers(X_train_raw, Y_train_raw, scaler_X, scaler_y, cfg)
    X_val_s, Y_val_s = apply_scalers(X_val_raw, Y_val_raw, scaler_X, scaler_y, cfg)
    X_test_s, Y_test_s = apply_scalers(X_test_raw, Y_test_raw, scaler_X, scaler_y, cfg)

    # Datasets
    train_ds = CFDPartitionDataset(X_train_s, Y_train_s)
    val_ds = CFDPartitionDataset(X_val_s, Y_val_s)
    test_ds = CFDPartitionDataset(X_test_s, Y_test_s)

    # DataLoaders
    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_per_gpu, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_per_gpu, sampler=val_sampler,
                            num_workers=cfg.num_workers, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=1, sampler=test_sampler,
                             num_workers=cfg.num_workers, pin_memory=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    # Keep raw test data for evaluation in physical units
    return (train_dl, val_dl, test_dl, scaler_X, scaler_y,
            Y_test_raw, meta_test,
            train_sampler if distributed else None)
