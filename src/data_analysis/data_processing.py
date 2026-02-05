import pandas as pd
import os
from typing import Optional, Tuple
from .config import SPLIT_SEED



def correct_sensor_bias(
    data: pd.DataFrame,
    bias_csv: str
) -> pd.DataFrame:
    """
    Apply per-sensor bias corrections to magnetic field measurements.

    Args:
        data: DataFrame with columns 'sensor_id', 'Bx', 'By', 'Bz'.
        bias_csv: Path to CSV file with columns 'sensor_id', 'Bx', 'By', 'Bz'.
    """

    df_bias = pd.read_csv(bias_csv)

    bx_bias = dict(zip(df_bias['sensor_id'], df_bias['Bx']))
    by_bias = dict(zip(df_bias['sensor_id'], df_bias['By']))
    bz_bias = dict(zip(df_bias['sensor_id'], df_bias['Bz']))

    df_bias_corrected = data.copy()
    df_bias_corrected['Bx'] = df_bias_corrected['Bx'] - df_bias_corrected['sensor_id'].map(bx_bias)
    df_bias_corrected['By'] = df_bias_corrected['By'] - df_bias_corrected['sensor_id'].map(by_bias)
    df_bias_corrected['Bz'] = df_bias_corrected['Bz'] - df_bias_corrected['sensor_id'].map(bz_bias)
    return df_bias_corrected


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    random_state: Optional[int] = SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Randomly split a DataFrame into train/validation/test DataFrames.

    Ratios should sum to 1.0 (or very close).
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df

def split_dataset_positional(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    resolution: float = 0.0016,  # meters (1.6 mm)
    shuffle: bool = True,
    random_state: Optional[int] = SPLIT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    df = df.copy()

    # Voxelize positions
    df[["x_key", "y_key", "z_key"]] = (df[["x", "y", "z"]] / resolution).round().astype(int)

    # Unique voxel positions
    pos_keys = df[["x_key", "y_key", "z_key"]].drop_duplicates()
    pos_keys = pos_keys.sort_values(["x_key", "y_key", "z_key"]).reset_index(drop=True)

    if shuffle:
        pos_keys = pos_keys.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n_pos = len(pos_keys)
    n_train = int(round(n_pos * train_ratio))
    n_val   = int(round(n_pos * val_ratio))
    # Ensure all positions are used: assign remainder to test
    n_test  = n_pos - n_train - n_val

    # In case rounding does something weird, clamp
    if n_test < 0:
        n_test = 0

    train_keys = pos_keys.iloc[:n_train]
    val_keys   = pos_keys.iloc[n_train:n_train + n_val]
    test_keys  = pos_keys.iloc[n_train + n_val:]

    # Helper to select rows by keys
    merge_cols = ["x_key", "y_key", "z_key"]
    train_df = df.merge(train_keys, on=merge_cols, how="inner")
    val_df   = df.merge(val_keys,   on=merge_cols, how="inner")
    test_df  = df.merge(test_keys,  on=merge_cols, how="inner")

    # Drop helper columns
    for d in (train_df, val_df, test_df):
        d.drop(columns=merge_cols, inplace=True)

    return train_df, val_df, test_df