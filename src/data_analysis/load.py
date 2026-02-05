import os
from pathlib import Path
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Union

# Pattern for filenames / paths like:
#   lvl0_pos0_rot0_base0_0_z3.3_20250926_134913.pkl
#   lvl0_pos0_rot1_base1_0_z0.0_20250926_143819.pkl
#   lvl0_pos0_rot0_base1_1_z0.0_20250927_142638.pkl
#
# We extract:
#   level      -> int
#   position   -> int
#   rotation   -> int
#   base_x     -> int  (the number after "base")
#   base_y     -> int  (the next number)
#   z          -> float
#
# The pattern is flexible thanks to '.*?' between sections, so it also works
# when these tokens appear in parent folders, or there are extra bits after 'z'.
_PATTERN = re.compile(
    r"lvl(?P<level>-?\d+).*?"           # lvl0
    r"pos(?P<position>-?\d+).*?"        # pos0
    r"rot(?P<rotation>-?\d+).*?"        # rot0
    r"base(?P<base_x>-?\d+)_?"          # base0_
    r"(?P<base_y>-?\d+).*?"             # 0_
    r"z(?P<z>-?\d+(?:\.\d+)?)",         # z3.3 or z0.0
    re.IGNORECASE,
)


def _extract_meta_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Apply the filename/path pattern to a string and return metadata dict
    or None if it doesn't match.
    """
    m = _PATTERN.search(text)
    if not m:
        return None
    d = m.groupdict()
    # Normalize types
    return {
        "level": int(d["level"]),
        "position": int(d["position"]),
        "rotation": int(d["rotation"]),
        "base_x": int(d["base_x"]),
        "base_y": int(d["base_y"]),
        "z": float(d["z"]),
    }


def _extract_meta_from_path(path: str) -> Optional[Dict[str, Any]]:
    """
    Try filename first, then include parent folders in case the info is there.
    """
    filename = os.path.basename(path)
    meta = _extract_meta_from_text(filename)
    if meta is not None:
        return meta

    # Fall back to searching the full path tokens (folders + filename)
    parts = []
    p = os.path.abspath(path)
    while True:
        p, tail = os.path.split(p)
        if tail:
            parts.append(tail)
        else:
            break
    token_str = "__".join(reversed(parts))  # keep path order
    return _extract_meta_from_text(token_str)


def _matches_filters(
    meta: Dict[str, Any],
    level,
    position,
    rotation,
    base_x,
    base_y,
    z,
) -> bool:
    """
    Check whether a meta dict matches the optional filter arguments.
    Each filter arg can be None (ignored) or a value comparable to the
    corresponding meta field (int/float/str).
    """

    def _ok(arg, value):
        if arg is None:
            return True
        # Allow str/int/float inputs; compare numerically when possible
        try:
            if isinstance(value, float):
                return float(arg) == value
            return int(arg) == int(value)
        except Exception:
            # Fallback to string compare
            return str(arg) == str(value)

    return (
        _ok(level,     meta["level"])
        and _ok(position, meta["position"])
        and _ok(rotation, meta["rotation"])
        and _ok(base_x,   meta["base_x"])
        and _ok(base_y,   meta["base_y"])
        and _ok(z,        meta["z"])
    )


def _load_file_to_df(fpath: str) -> pd.DataFrame:
    """
    Load a file into a DataFrame.
    - .csv -> pd.read_csv
    - .pkl -> expects dict with keys {'header','rows'} and builds a DataFrame.
              If it's already a DataFrame (or list[dict]), it will be coerced accordingly.
    """
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(fpath)

    # .pkl
    obj = pd.read_pickle(fpath)
    if isinstance(obj, dict) and "header" in obj and "rows" in obj:
        header = obj["header"]
        rows = obj["rows"]
        # rows can be list[list] or list[tuple]
        df = pd.DataFrame(rows, columns=list(header))

        return df
    # Fallbacks: if the pickle is already a DataFrame, or list[dict], etc.
    if isinstance(obj, pd.DataFrame):
        return obj
    try:
        return pd.DataFrame(obj)  # best-effort coercion
    except Exception as e:
        raise ValueError(f"Unsupported pickle content in {fpath}: {type(obj)}") from e


def _attach_meta_cols(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Add metadata columns to a copy of df:
      - lvl, pos, rot
      - base_x, base_y
      - z_offset
    """
    out = df.copy()
    out["lvl"]       = int(meta["level"])
    out["pos"]       = int(meta["position"])
    out["rot"]       = int(meta["rotation"])
    out["base_x"]    = int(meta["base_x"])
    out["base_y"]    = int(meta["base_y"])
    out["z_offset"]  = float(meta["z"])
    return out

def load_data(
    dir: str,
    concat: bool = True,
    *,
    level=None,
    position=None,
    rotation=None,
    base_x=None,
    base_y=None,
    z=None,
    ignore_file: Optional[str] = None,
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Walk `dir`, load .csv/.pkl files whose names (and/or parent folders) match:
      lvl{level}_pos{position}_rot{rotation}_base{base_x}_{base_y}_z{z}_<timestamp>.(csv|pkl)

    If `ignore_file` is provided, any filename or relative path listed (one per line)
    will be skipped.
    """
    # Build ignore set (filenames or relative paths), ignoring empty lines and comments
    ignore_names: set[str] = set()
    if ignore_file is not None:
        if not os.path.isfile(ignore_file):
            raise FileNotFoundError(f"ignore_file not found: {ignore_file}")
        with open(ignore_file, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#"):
                    ignore_names.add(s)

    records: List[Dict[str, Any]] = []
    dir = os.path.abspath(dir)

    for root, _, files in os.walk(dir): 
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in {".csv", ".pkl"}:
                continue

            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, dir)

            # Skip if listed in ignore file (match by filename or rel path)
            if ignore_names:
                if fname in ignore_names or relpath in ignore_names:
                    print(f'Skipping "{relpath}" (in ignore list).')
                    continue

            meta = _extract_meta_from_path(fpath)
            if meta is None:
                continue

            if not _matches_filters(
                meta,
                level,
                position,
                rotation,
                base_x,
                base_y,
                z,
            ):
                continue

            print(f'Loading "{fname}"...')

            df = _load_file_to_df(fpath)
            df = _attach_meta_cols(df, meta)

            # Ensure sensor_id is integer if present
            if "sensor_id" in df.columns:
                df["sensor_id"] = df["sensor_id"].astype(int)

            records.append({"path": fpath, "meta": meta, "data": df})

    records.sort(
        key=lambda r: (
            r["meta"]["level"],
            r["meta"]["position"],
            r["meta"]["rotation"],
            r["meta"]["base_x"],
            r["meta"]["base_y"],
            r["meta"]["z"],
            r["path"],
        )
    )

    if not concat:
        return records

    if not records:
        return pd.DataFrame()

    try:
        return pd.concat([r["data"] for r in records], ignore_index=True, sort=False)
    except Exception as e:
        raise RuntimeError(
            "Failed to concatenate loaded DataFrames. "
            "Try concat=False to inspect per-file schemas."
        ) from e
    


def load_pkl_paths(original_data_dir, ignore_file=None):
    data_dir = os.path.abspath(original_data_dir)
    ignore_file = os.path.abspath(ignore_file) if ignore_file else None

    # Load ignore list
    ignore_set = set()
    ignore_set_lower = set()
    if ignore_file and os.path.isfile(ignore_file):
        with open(ignore_file, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                norm = os.path.normpath(s)
                ignore_set.add(s)
                ignore_set.add(norm)
                ignore_set_lower.add(s.lower())
                ignore_set_lower.add(norm.lower())

    files = []
    ignored_files = []

    # Collect all candidate .pkl paths
    for root, _, fs in os.walk(data_dir):
        for fname in fs:
            if not fname.lower().endswith(".pkl"):
                continue

            src_path = os.path.join(root, fname)
            rel_path = os.path.relpath(src_path, data_dir)
            norm_rel = os.path.normpath(rel_path)

            # Apply ignore filters
            if (
                fname in ignore_set
                or rel_path in ignore_set
                or norm_rel in ignore_set
                or fname.lower() in ignore_set_lower
                or rel_path.lower() in ignore_set_lower
                or norm_rel.lower() in ignore_set_lower
            ):
                ignored_files.append(src_path)
                continue

            files.append(src_path)

    return files, ignored_files


def load_pkl_to_df(path):
    obj = pd.read_pickle(path)

    if isinstance(obj, dict) and "header" in obj and "rows" in obj:
        header = obj["header"]
        rows = obj["rows"]
        return pd.DataFrame(rows, columns=list(header))
    
    raise ValueError(f"Unrecognized pickle format in file: {path}")

def load_raw_set_of_pkls(data_dir, ignore_file=None):
    data_dir = Path(data_dir)

    print(f"Loading .pkl files from: {data_dir}")

    # Build ignore set
    ignore_set = set()
    if ignore_file is not None:
        ignore_file = Path(ignore_file)
        lines = ignore_file.read_text().splitlines()
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = Path(s)
            if not p.is_absolute():
                p = (data_dir / p).resolve()
            else:
                p = p.resolve()
            ignore_set.add(p)

    # Find pkls
    pkl_paths = sorted([p for p in data_dir.rglob("*.pkl") if p.is_file()])
    pkl_paths = [p for p in pkl_paths if p.resolve() not in ignore_set]

    if not pkl_paths:
        raise FileNotFoundError(f"No .pkl files found under: {data_dir}")

    dfs = []
    ref_cols = None
    ref_path = None

    for p in pkl_paths:
        df = pd.read_pickle(p)

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Pickle is not a pandas DataFrame: {p}")

        cols = list(df.columns)
        if ref_cols is None:
            ref_cols = cols
            ref_path = p
        else:
            if cols != ref_cols:
                # also show set diffs to help debug
                extra = sorted(set(cols) - set(ref_cols))
                missing = sorted(set(ref_cols) - set(cols))
                raise ValueError(
                    "Columns mismatch.\n"
                    f"Reference: {ref_path}\n"
                    f"Offender:  {p}\n"
                    f"Missing columns in offender: {missing}\n"
                    f"Extra columns in offender:   {extra}\n"
                    "Note: column order must match too."
                )

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    return out


def load_octomag_format(
    dir: str,
    concat: bool = True,
    *,
    level=None,
    position=None,
    rotation=None,
    base_x=None,
    base_y=None,
    z=None,
    ignore_file: Optional[str] = None,
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    return load_data(
        dir,
        concat=concat,
        level=level,
        position=position,
        rotation=rotation,
        base_x=base_x,
        base_y=base_y,
        z=z,
        ignore_file=ignore_file,
    )

def load_navion_format(data_dir, ignore_file=None):
    df = load_raw_set_of_pkls(data_dir, ignore_file=ignore_file)
    # Convert from Tesla to militesla
    df[["Bx", "By", "Bz"]] = df[["Bx", "By", "Bz"]] * 1e3
    return df

def apply_navion_transform(df):
    df['x_trans'] = -df['x']
    df['y_trans'] = -df['y'] + 0.2
    df['z_trans'] = df['z']
    return df