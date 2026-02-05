import re
from typing import Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
import os, json, zipfile
from abc import ABC

from .constants import OCTOMAG_EMS
from .calibration import Calibration

class GBTCalibration(Calibration):
    def __init__(self, 
                 name: str = "GBT_CALIBRATION",
                 target_names: list = ["Bx", "By", "Bz"],
                 position_names: list = ["x", "y", "z"],
                 current_names: list = OCTOMAG_EMS):

        super().__init__(name, len(current_names))
        self._core = None
        self._params = None
        self._target_normalization_dict = None
        self._position_normalization_dict = None

        self._leaf_tables = None # For differentiation of linear leaves

        self._target_names = target_names
        self._position_names = position_names
        self._current_names = current_names if current_names is not None else []

        self._features_ordered = self._position_names + self._current_names

        # --- speed caches ---
        self._feat_to_idx = {f: i for i, f in enumerate(self._features_ordered)}
        # cache dense per-tree coefficient matrices for a given (target, wrt_features)
        self._leaf_w_mats_cache = {}

    @property
    def core(self):
        return self._core
    @property
    def target_names(self):
        return self._target_names
    @property
    def position_names(self):
        return self._position_names
    @property
    def current_names(self):
        return self._current_names
    @property
    def target_normalization_dict(self) -> Optional[dict]:
        return self._target_normalization_dict
    @property
    def position_normalization_dict(self) -> Optional[dict]:
        return self._position_normalization_dict
    
    def predict_targets(self, position: np.ndarray, currents: Optional[np.ndarray] = None) -> np.ndarray:
        self._check_model_trained()
        position, currents, batched = self._check_input(position, currents)

        X = self._build_input_array(position, currents)

        preds = []
        for t in self._target_names:
            m = self._core[t]

            if hasattr(m, "booster_"):  # LGBMRegressor
                it = self._num_iteration_for_target(t, m)
                pred_t = m.predict(X, num_iteration=it)
            else:  # Booster
                it = self._num_iteration_for_target(t, m)
                pred_t = m.predict(X, num_iteration=it)

            preds.append(np.asarray(pred_t).reshape(-1, 1))

        preds_np = np.hstack(preds)

        # Denormalize if needed
        if self._target_normalization_dict is not None:
            mean = self._target_normalization_dict["mean"]
            std = self._target_normalization_dict["std"]
            preds_np = preds_np * std + mean
        
        return preds_np if batched else preds_np.flatten()
    
    def compute_grads(self, models, X, wrt_features):

        # If wrt_features is position names, but with different order, throw warning stating that position denormalization will be not be done
        if set(wrt_features) == set(self._position_names) and wrt_features != self._position_names:
            print("Warning: wrt_features matches position names but with different order. Position denormalization will not be applied.")

        target_norm_params = self._target_normalization_dict
        input_norm_params = self._position_normalization_dict if wrt_features == self._position_names else None

        feature_cols = self._features_ordered
        target_cols = self._target_names

        # Precompute leaf tables if not done yet
        if self._leaf_tables is None:
            self._leaf_tables = __class__._compute_leaf_tables(models, feature_cols)
        leaf_tables_by_target = self._leaf_tables

        # ensure X is in the right order once
        if isinstance(X, pd.DataFrame):
            X_use = X[feature_cols]
        else:
            X_use = np.asarray(X)

        grads_dict = {}
        grads_stack = np.zeros((len(X_use), len(target_cols), len(wrt_features)), dtype=float)

        for field_index, t in enumerate(target_cols):
            m = models[t]
            booster = m.booster_ if hasattr(m, "booster_") else m
            
            # Make X contiguous float32 once
            if isinstance(X_use, pd.DataFrame):
                X_mat = X_use[feature_cols].to_numpy()
            else:
                X_mat = np.asarray(X_use)
            X_mat = np.ascontiguousarray(X_mat, dtype=np.float32)

            # Respect best_iteration in leaf prediction
            it = self._num_iteration_for_target(t, m if hasattr(m, "booster_") else booster)
            leaf_idx = booster.predict(X_mat, pred_leaf=True, num_iteration=it).astype(np.int32)
            n_samples, n_trees = leaf_idx.shape

            leaf_tables = leaf_tables_by_target[t][:n_trees]

            # Cached dense (leaf -> wrt-coefs) matrices per tree
            leaf_w_mats = self._get_leaf_w_mats(t, wrt_features, leaf_tables)

            # Accumulate gradients: only gather + add (no per-leaf building)
            grads_norm = np.zeros((n_samples, len(wrt_features)), dtype=np.float32)
            for tree_i in range(n_trees):
                mat = leaf_w_mats[tree_i]
                if mat is None:
                    continue
                grads_norm += mat[leaf_idx[:, tree_i], :]
            
            grads = __class__._denorm_grads(grads_norm, target_norm_params, input_norm_params, field_index)

            grads_dict[t] = grads
            grads_stack[:, field_index, :] = grads

        return grads_dict, grads_stack
    
    def compute_grads_wrt_position(self, position: np.ndarray, currents: Optional[np.ndarray] = None):
        self._check_model_trained()
        position, currents, batched = self._check_input(position, currents)

        X = self._build_input_array(position, currents)

        _grads_dict, J = self.compute_grads(self._core, X, self._position_names)

        return J if batched else J[0]
    
    def compute_grads_wrt_currents(self, position: np.ndarray, currents: Optional[np.ndarray] = None):
        self._check_model_trained()
        position, currents, batched = self._check_input(position, currents)

        # Build input array
        X = self._build_input_array(position, currents)

        _grads_dict, J = self.compute_grads(self._core, X, self._current_names)

        return J if batched else J[0]
    
    def train(self,
            # Data
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,

            # Should we normalize?
            normalize_targets: bool = True,
            normalize_position: bool = False,

            # GBT hyperparameters
            n_estimators: int = 5000,
            learning_rate: float = 0.1,
            num_leaves: int = 32,
            min_child_samples: int = 20,
            subsample: float = 1.0,
            colsample_bytree: float = 1.0,
            random_state: int = 3,

            # Early stopping
            early_stopping_patience: int = 200,
            
            # Bothering
            verbose: bool = True):
        
        """Train GBT calibration model."""

        # Check if target, position, current names are in dataframes
        for name in self._target_names + self._position_names + self._current_names:
            if name not in train_df.columns:
                raise ValueError(f"Column '{name}' not found in train_df")
            if name not in val_df.columns:
                raise ValueError(f"Column '{name}' not found in val_df")
            
        # Normalize if needed
        normalized_train_df = train_df.copy()
        normalized_val_df = val_df.copy()

        if normalize_targets:
            target_mean = train_df[self._target_names].mean().to_numpy()
            target_std = train_df[self._target_names].std().to_numpy()

            self._target_normalization_dict = {
                "mean": target_mean,
                "std": target_std
            }

            normalized_train_df[self._target_names] = (train_df[self._target_names] - target_mean) / target_std
            normalized_val_df[self._target_names] = (val_df[self._target_names] - target_mean) / target_std

        if normalize_position:
            position_mean = train_df[self._position_names].mean().to_numpy()
            position_std = train_df[self._position_names].std().to_numpy()

            self._position_normalization_dict = {
                "mean": position_mean,
                "std": position_std
            }

            normalized_train_df[self._position_names] = (train_df[self._position_names] - position_mean) / position_std
            normalized_val_df[self._position_names] = (val_df[self._position_names] - position_mean) / position_std

        # Prepare np datasets
        X_train = normalized_train_df[self._features_ordered].to_numpy()
        y_train = {target_name: normalized_train_df[target_name].to_numpy() for target_name in self._target_names}
        X_val = normalized_val_df[self._features_ordered].to_numpy()
        y_val = {target_name: normalized_val_df[target_name].to_numpy() for target_name in self._target_names}

        # Build params dict
        self._params = dict(
            objective = "regression",
            n_estimators = n_estimators,
            learning_rate = learning_rate,
            num_leaves = num_leaves,
            min_child_samples = min_child_samples,
            subsample = subsample,
            colsample_bytree = colsample_bytree,
            random_state = random_state,
            reg_lambda = 0.0,

            linear_tree = True,
            tree_learner = "serial",
            device_type = "cpu",

            n_jobs = -1, # Use all cores

            seed=random_state,
        )

        # Helper so that physical rmse is reported during training
        def make_denorm_rmse(std, name="rmse_denorm"):
            std = float(std)
            def _metric(y_true, y_pred):
                rmse_norm = np.sqrt(np.mean((y_true - y_pred) ** 2))
                rmse_orig = rmse_norm * std
                return (name, rmse_orig, False)  # lower is better
            return _metric



        # Train one model per target
        models = {}
        best_iters = {}
        eval_hist = {}          # target -> history dict
        best_rmse = {}          # target -> rmse at best iter (denorm)

        for i, t in enumerate(self._target_names):
            print(f"\nTraining target: {t}")

            metric_name = f"{t}_rmse"
            denorm_rmse = make_denorm_rmse(self._target_normalization_dict["std"][i], name=metric_name)

            model = lgb.LGBMRegressor(**self._params)

            history = {}  # LightGBM will fill this
            model.fit(
                X_train, y_train[t],
                eval_set=[(X_val, y_val[t])],
                eval_metric=[denorm_rmse],
                callbacks=[
                    lgb.record_evaluation(history),
                    lgb.early_stopping(early_stopping_patience, first_metric_only=True, verbose=verbose),
                    lgb.log_evaluation(period=50),
                ],
            )

            models[t] = model
            eval_hist[t] = history
            best_iters[t] = model.best_iteration_
            
            bi = best_iters[t]
            best_rmse[t] = history["valid_0"][metric_name][bi - 1]

        print("\nBest iterations:", best_iters)
        print("Best (denorm) RMSE at best iteration:", best_rmse)

        # Keep the trained models
        self._core = models

        self._leaf_tables = None
        self._leaf_w_mats_cache = {}

        self._best_iteration_by_target = best_iters


    # Loading and saving
    def save(self, path: str):
        """
        Save EVERYTHING into a single file (zip bundle).

        If `path` is a directory, we save:
            <path>/<self.name>_YYYYMMDD_HHMMSS.gbt.zip
        If `path` is a file path, we save exactly there.
        """
        self._check_model_trained()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Decide output file path
        if os.path.isdir(path) or path.endswith(os.sep):
            fname = f"{self._name}_{timestamp}.gbt.zip"
            out_path = os.path.join(path, fname)
        else:
            out_path = path  # user provided full filename

        def _jsonify_norm(d):
            if d is None:
                return None
            return {
                "mean": np.asarray(d["mean"]).tolist(),
                "std":  np.asarray(d["std"]).tolist(),
            }

        meta = {
            "name": self._name,
            "target_names": list(self._target_names),
            "position_names": list(self._position_names),
            "current_names": list(self._current_names) if self._current_names is not None else None,
            "features_ordered": list(self._features_ordered),
            "params": self._params,
            "target_normalization_dict": _jsonify_norm(self._target_normalization_dict),
            "position_normalization_dict": _jsonify_norm(self._position_normalization_dict),
            "lightgbm_version": getattr(lgb, "__version__", None),
            "saved_at": timestamp,
            "best_iteration_by_target": {k: int(v) for k, v in (getattr(self, "_best_iteration_by_target", {}) or {}).items()},
        }

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps(meta, indent=2))

            for t, booster_or_model in self._core.items():
                booster = booster_or_model.booster_ if hasattr(booster_or_model, "booster_") else booster_or_model
                zf.writestr(f"model_{t}.txt", booster.model_to_string())

        return out_path
    

    @classmethod
    def load_from(cls, path: str):
        import os, json, zipfile
        import numpy as np
        import lightgbm as lgb

        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        with zipfile.ZipFile(path, "r") as zf:
            meta = json.loads(zf.read("meta.json").decode("utf-8"))

            # --- construct instance ---
            try:
                obj = cls(
                    name=meta.get("name", "GBT_CALIBRATION"),
                    target_names=meta["target_names"],
                    position_names=meta["position_names"],
                    current_names=meta.get("current_names", None),
                )
            except TypeError:
                # e.g. DirectGBT(name=..., current_names=...) only
                obj = cls(
                    name=meta.get("name", "GBT_CALIBRATION"),
                    current_names=meta.get("current_names", None),
                )
                # still restore these (important for prediction / grads)
                obj._target_names = meta["target_names"]
                obj._position_names = meta["position_names"]

            # restore basic attrs
            obj._features_ordered = meta.get("features_ordered", obj._features_ordered)
            obj._params = meta.get("params", None)

            # restore normalization dicts
            def _numpyfy_norm(d):
                if d is None:
                    return None
                return {
                    "mean": np.asarray(d["mean"], dtype=float),
                    "std":  np.asarray(d["std"], dtype=float),
                }

            obj._target_normalization_dict = _numpyfy_norm(meta.get("target_normalization_dict", None))
            obj._position_normalization_dict = _numpyfy_norm(meta.get("position_normalization_dict", None))

            # load boosters
            obj._core = {}
            for t in obj._target_names:
                model_txt = zf.read(f"model_{t}.txt").decode("utf-8")
                obj._core[t] = lgb.Booster(model_str=model_txt)

            # leaf tables cache should be rebuilt lazily
            obj._leaf_tables = None


            obj._leaf_w_mats_cache = {}
            obj._feat_to_idx = {f: i for i, f in enumerate(obj._features_ordered)}

            obj._best_iteration_by_target = meta.get("best_iteration_by_target", None) or {}


            return obj

    def _check_model_trained(self):
        if self._core is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
    def _check_input(self, positions: np.ndarray, currents: Optional[np.ndarray] = None):
        if self._current_names is not None and currents is None:
            raise ValueError("currents must be provided since current_names is not None.")
        elif self._current_names is None and currents is not None:
            raise ValueError("currents should be None since current_names is None.")
        # Make sure inputs are 2D arrays    
        batched = True
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
            batched = False
        if currents is not None and currents.ndim == 1:
            currents = currents.reshape(1, -1)
            batched = False

        # Check shapes
        if positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")
        if currents is not None and currents.shape[1] != len(self._current_names):
            raise ValueError(f"currents must have shape (N, {len(self._current_names)}), got {currents.shape}")
        if currents is not None and positions.shape[0] != currents.shape[0]:
            raise ValueError(f"positions and currents must have the same number of samples, got {positions.shape[0]} and {currents.shape[0]}")
        
        return positions, currents, batched

    def _build_input_array(self, position: np.ndarray, currents: Optional[np.ndarray] = None) -> np.ndarray:
        """Build input array X from position and currents."""
        # NOTE: It is crucial that the order of features matches self._features_ordered!!!!!!
        if currents is not None:
            X = np.hstack([position, currents])
        else:
            X = position
        return X

    # Methods for differentiation
    @staticmethod
    def _parse_block_keyvals(lines):
        """Parse key=value blocks where values can continue onto following lines."""
        _key_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")
        out = {}
        i = 0
        while i < len(lines):
            m = _key_re.match(lines[i])
            if not m:
                i += 1
                continue
            key = m.group(1)
            rest = lines[i][len(key) + 1 :].strip()
            toks = rest.split() if rest else []
            i += 1
            while i < len(lines) and not _key_re.match(lines[i]):
                toks += lines[i].strip().split()
                i += 1
            out[key] = toks
        return out

    @staticmethod
    def _build_linear_leaf_tables(booster, n_features):
        """
        Returns list leaf_tables where:
        leaf_tables[tree_idx][leaf_idx] = (const, coeff_vec)
        coeff_vec has length n_features in the model's feature order.

        NOTE: Uses booster.model_to_string() because dump_model() JSON in LightGBM 4.6.0
        does not expose leaf_coeff/leaf_features/leaf_const for linear_tree.
        """
        s = booster.model_to_string()
        lines = s.splitlines()

        # find tree blocks
        tree_starts = [i for i, ln in enumerate(lines) if ln.startswith("Tree=")]
        tree_starts.append(len(lines))

        tables = []

        for a, b in zip(tree_starts[:-1], tree_starts[1:]):
            block = lines[a:b]
            kv = __class__._parse_block_keyvals(block)

            # If no linear info in this tree, treat as constant-leaf => zero gradient
            if ("leaf_coeff" not in kv) or ("leaf_features" not in kv) or ("leaf_const" not in kv) or ("num_features" not in kv):
                tables.append({})
                continue

            num_leaves = int(kv["num_leaves"][0])
            num_features = list(map(int, kv["num_features"]))     # length = num_leaves
            leaf_const   = list(map(float, kv["leaf_const"]))     # length = num_leaves
            feat_flat    = list(map(int, kv["leaf_features"]))    # length = sum(num_features)
            coef_flat    = list(map(float, kv["leaf_coeff"]))     # length = sum(num_features)

            leaf_table = {}
            off = 0
            for leaf_id in range(num_leaves):
                k = num_features[leaf_id]
                feats = feat_flat[off:off+k]
                coefs = coef_flat[off:off+k]
                off += k

                w = np.zeros(n_features, dtype=float)
                for f, c in zip(feats, coefs):
                    if 0 <= f < n_features:
                        w[int(f)] = float(c)

                leaf_table[leaf_id] = (float(leaf_const[leaf_id]), w)

            tables.append(leaf_table)

        return tables
    
    @staticmethod
    def _denorm_grads(grads, output_normalization_params, input_normalization_params, field_index):
        grads = np.asarray(grads, dtype=float)

        output_std = output_normalization_params["std"][field_index] if output_normalization_params else 1.0  # scalar
        input_std = input_normalization_params["std"] if input_normalization_params is not None else [1.0 for _ in range(grads.shape[1])]

        input_std = np.asarray(input_std, dtype=float)  # shape (3,)
        return grads * output_std / input_std
    
    @staticmethod
    def _compute_leaf_tables(models, feature_cols):
        n_features = len(feature_cols)
        return {
                t: __class__._build_linear_leaf_tables(
                    (models[t].booster_ if hasattr(models[t], "booster_") else models[t]),
                    n_features=n_features
                )
                for t in models.keys()
                }


    @staticmethod
    def _effective_num_iteration(booster_or_model):
        # Works for LGBMRegressor or Booster
        booster = booster_or_model.booster_ if hasattr(booster_or_model, "booster_") else booster_or_model

        # best_iteration_ on sklearn wrapper, best_iteration on Booster
        it = getattr(booster_or_model, "best_iteration_", None)
        if it is None:
            it = getattr(booster, "best_iteration", 0) or 0

        if it and it > 0:
            return int(it)

        # fallback: use all current trees
        cur = getattr(booster, "current_iteration", None)
        return int(cur()) if callable(cur) else None
    
    def _get_leaf_w_mats(self, target: str, wrt_features: list, leaf_tables: list):
        """
        Returns list of matrices per tree:
          mats[tree_i] is (n_leaves_in_tree, len(wrt_features)) float32
          or None if that tree has no linear info.
        Cached by (target, tuple(wrt_features)).
        """
        key = (target, tuple(wrt_features))
        cached = self._leaf_w_mats_cache.get(key, None)
        if cached is not None:
            return cached

        wrt_idx = np.array([self._feat_to_idx[f] for f in wrt_features], dtype=np.int32)

        mats = []
        for table in leaf_tables:
            if not table:
                mats.append(None)
                continue

            n_leaves = max(table.keys()) + 1
            mat = np.zeros((n_leaves, len(wrt_idx)), dtype=np.float32)

            # fill once per tree
            for li, (_, wfull) in table.items():
                mat[li, :] = wfull[wrt_idx]

            mats.append(mat)

        self._leaf_w_mats_cache[key] = mats
        return mats
    
    def _num_iteration_for_target(self, t: str, model_or_booster):
        """
        Returns the num_iteration to use for prediction / pred_leaf for target t.
        - If sklearn wrapper: use best_iteration_
        - If Booster after load: use saved best_iteration_by_target if available
        - Else: fall back to Booster.best_iteration if set
        - Else: None (use all trees)
        """
        # sklearn wrapper
        if hasattr(model_or_booster, "best_iteration_"):
            it = getattr(model_or_booster, "best_iteration_", 0) or 0
            return int(it) if it > 0 else None

        booster = model_or_booster  # Booster
        # prefer saved best iters (per-target)
        d = getattr(self, "_best_iteration_by_target", None)
        if isinstance(d, dict) and t in d and d[t]:
            it = int(d[t])
            return it if it > 0 else None

        # fallback to booster attribute
        it = getattr(booster, "best_iteration", 0) or 0
        return int(it) if it > 0 else None