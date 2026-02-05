import os

import numpy as np
from calibration import Calibration
from .phantom import ModelPhantom
from tqdm import tqdm
import warnings
import pickle


class EvaluationPackage:
    def __init__(self):
        # Models
        self._models = {}
        self._phantoms = []

        # Features/Labels
        self._positions = {}  # {pct: np.ndarray}
        self._currents  = {}  # {pct: np.ndarray}
        self._fields    = {}  # {pct: np.ndarray}

        # Predictions
        self._field_prediction_dict = None
        self._gradient_prediction_dict = None

        # Metrics
        self._field_metrics_dict = None
        self._gradient_metrics_dict = None

        # Gradient references from polynomial fits
        self._gradient_reference_dict = None
        self._chosen_order = None
        self._chosen_neighbor_count = None

    @property
    def models(self):
        return self._models
    @property
    def phantoms(self):
        return self._phantoms
    @property
    def positions(self):
        return self._positions
    @property
    def currents(self):
        return self._currents
    @property
    def fields(self):
        return self._fields
    @property
    def field_predictions(self):
        return self._field_prediction_dict
    @property
    def gradient_predictions(self):
        return self._gradient_prediction_dict
    @property
    def field_metrics(self):
        return self._field_metrics_dict
    @property
    def gradient_metrics(self):
        return self._gradient_metrics_dict
    
    def print_phantoms(self):
        for phantom in self._phantoms:
            print(phantom.repr())

    def get_field_predictions(self, phantoms):
        single = False
        if type(phantoms) is not list:
            phantoms = [phantoms]
            single = True

        out = {}
        for phantom in phantoms:
            base_name, dataset_percentage, structure = phantom.keys()
            try:
                preds = self._field_prediction_dict[base_name][dataset_percentage][structure]
            except Exception:
                raise KeyError(f"Missing field predictions for model '{phantom.string(verbose=True)}'")

            out[phantom.string(verbose=True)] = preds

        if single:
            return out[phantoms[0].string(verbose=True)]

        return out
    
    def get_gradient_predictions(self, phantoms):
        single = False
        if type(phantoms) is not list:
            phantoms = [phantoms]
            single = True

        out = {}
        for phantom in phantoms:
            base_name, dataset_percentage, structure = phantom.keys()
            try:
                preds = self._gradient_prediction_dict[base_name][dataset_percentage][structure]
            except Exception:
                raise KeyError(f"Missing gradient predictions for model '{phantom.string(verbose=True)}'")

            out[phantom.string(verbose=True)] = preds

        if single:
            return out[phantoms[0].string(verbose=True)]

        return out


    def get_field_metrics(self, phantoms):
        single = False
        if type(phantoms) is not list:
            phantoms = [phantoms]
            single = True

        out = {}
        for phantom in phantoms:
            base_name, dataset_percentage, structure = phantom.keys()
            try:
                metrics = self._field_metrics_dict[base_name][dataset_percentage][structure]
            except Exception:
                raise KeyError(f"Missing field metrics for model '{phantom.string(verbose=True)}'")

            out[phantom.string(verbose=True)] = metrics

        if single:
            return out[phantoms[0].string(verbose=True)]

        return out


    def get_gradient_metrics(self, phantoms):
        single = False
        if type(phantoms) is not list:
            phantoms = [phantoms]
            single = True

        out = {}
        for phantom in phantoms:
            base_name, dataset_percentage, structure = phantom.keys()
            try:
                metrics = self._gradient_metrics_dict[base_name][dataset_percentage][structure]
            except Exception:
                raise KeyError(f"Missing gradient metrics for model '{phantom.string(verbose=True)}'")

            out[phantom.string(verbose=True)] = metrics

        if single:
            return out[phantoms[0].string(verbose=True)]

        return out



    def load_model(self, model, force=False):
        """
        Store model at:
        self._models[base_name][dataset_percentage][structure] = model

        dataset_percentage is an int (0..100)
        structure is tuple(...) for nets/gbt, or None for MPEM (or anything without structure)
        """
        if not isinstance(model, Calibration):
            raise ValueError("load_model expects a Calibration (or subclass) instance.")

        phantom = ModelPhantom.from_calibration(model)

        # Check if we already have this model
        if phantom.keys() in [p.keys() for p in self._phantoms]:
            if not force:
                warnings.warn(
                    f"Model '{phantom.string(verbose=True)}' is already loaded. "
                    "Use force=True to overwrite."
                )
                return
            else:
                warnings.warn(f"Overwriting existing model '{phantom.string(verbose=True)}'.")

                # Remove existing phantom (clarity)
                self._phantoms = [p for p in self._phantoms if p.keys() != phantom.keys()]

        self._phantoms.append(phantom)

        base_name, dataset_percentage, structure = phantom.keys()  # (str, int, tuple|None)

        level1 = self._models.setdefault(base_name, {})
        level2 = level1.setdefault(dataset_percentage, {})
        level2[structure] = model

        # remove old predictions/metrics if present
        if self._field_prediction_dict is not None:
            _pop_nested(self._field_prediction_dict, base_name, dataset_percentage, structure, default=None)

        if self._field_metrics_dict is not None:
            _pop_nested(self._field_metrics_dict, base_name, dataset_percentage, structure, default=None)

        if self._gradient_prediction_dict is not None:
            _pop_nested(self._gradient_prediction_dict, base_name, dataset_percentage, structure, default=None)

        if self._gradient_metrics_dict is not None:
            _pop_nested(self._gradient_metrics_dict, base_name, dataset_percentage, structure, default=None)
    
    def load_features_and_labels(self, positions, currents, fields, dataset_percentage=100, overwrite=False):
        pct = int(dataset_percentage)
        if pct < 0 or pct > 100:
            raise ValueError("dataset_percentage must be between 0 and 100.")

        exists = (pct in self._positions) or (pct in self._currents) or (pct in self._fields)

        # If any exist, require all exist (guard against partial state)
        if exists and not ((pct in self._positions) and (pct in self._currents) and (pct in self._fields)):
            raise ValueError(
                f"Partial labels already loaded for dataset_percentage={pct}. "
                "Cannot safely overwrite/compare. Fix internal state first."
            )

        if exists and not overwrite:
            raise ValueError(
                f"Features/labels for dataset_percentage={pct} already exist. "
                "Pass overwrite=True to replace them."
            )

        # Store (new or overwrite)
        self._positions[pct] = positions
        self._currents[pct]  = currents
        self._fields[pct]    = fields

        print(f"Loaded features and labels for dataset_percentage={pct} successfully.")

    def load_gradient_references(self, gradient_reference_dict, chosen_order=None, chosen_neighbor_count=None):
        full_pct = int(100)

        # Ensure full dataset labels are loaded
        if (full_pct not in self.positions) or (full_pct not in self.currents) or (full_pct not in self.fields):
            raise ValueError(f"Full dataset labels (dataset_percentage={full_pct}) must be loaded before loading gradient references.")

        # Get positions from dict. If not present, throw error.
        pos_key = "positions"
        if pos_key not in gradient_reference_dict:
            raise ValueError(f"Gradient reference dict must contain key '{pos_key}' with positions.")
        grad_positions = gradient_reference_dict[pos_key]
        if not np.array_equal(grad_positions, self.positions[full_pct]):
            raise ValueError("Gradient reference positions do not match loaded FULL positions.")

        # Same for currents and fields
        curr_key = "currents"
        if curr_key not in gradient_reference_dict:
            raise ValueError(f"Gradient reference dict must contain key '{curr_key}' with currents.")
        grad_currents = gradient_reference_dict[curr_key]
        if not np.array_equal(grad_currents, self.currents[full_pct]):
            raise ValueError("Gradient reference currents do not match loaded FULL currents.")

        field_key = "fields"
        if field_key not in gradient_reference_dict:
            raise ValueError(f"Gradient reference dict must contain key '{field_key}' with fields.")
        grad_fields = gradient_reference_dict[field_key]
        if not np.array_equal(grad_fields, self.fields[full_pct]):
            raise ValueError("Gradient reference fields do not match loaded FULL fields.")

        # Store full reference dict only
        self._gradient_reference_dict = gradient_reference_dict
        self._chosen_order = chosen_order
        self._chosen_neighbor_count = chosen_neighbor_count

        print(f"Loaded gradient reference dict successfully (validated against dataset_percentage={full_pct}).")

    def set_chosen_gradient_reference_params(self, chosen_order, chosen_neighbor_count):
        self._chosen_order = chosen_order
        self._chosen_neighbor_count = chosen_neighbor_count

    def manual_load_field_predictions(self, phantom, field_predictions):
        base_name, dataset_percentage, structure = phantom.keys()

        if self._field_prediction_dict is None:
            self._field_prediction_dict = {}
        level1 = self._field_prediction_dict.setdefault(base_name, {})
        level2 = level1.setdefault(dataset_percentage, {})

        level2[structure] = field_predictions

        # Add to phantoms if not already present
        if phantom.keys() not in [p.keys() for p in self._phantoms]:
            self._phantoms.append(phantom)

    def manual_load_gradient_predictions(self, phantom, gradient_predictions):
        base_name, dataset_percentage, structure = phantom.keys()

        if self._gradient_prediction_dict is None:
            self._gradient_prediction_dict = {}
        level1 = self._gradient_prediction_dict.setdefault(base_name, {})
        level2 = level1.setdefault(dataset_percentage, {})

        level2[structure] = gradient_predictions

        # Add to phantoms if not already present
        if phantom.keys() not in [p.keys() for p in self._phantoms]:
            self._phantoms.append(phantom)

    def compute_field_predictions(self, keep_existing=True, skip_models=None):
        """
        Compute and store field predictions for all loaded phantoms.

        Stores at:
            self._field_prediction_dict[base_name][dataset_percentage][structure] = np.ndarray

        Parameters
        ----------
        keep_existing : bool
            If True, do not overwrite already-computed predictions.
        skip_models : list[ModelPhantom] | None
            Optional list of ModelPhantom to skip (compared by phantom.keys()).
        """
        if not self._positions or not self._currents:
            raise ValueError("Positions and currents must be loaded before computing predictions.")

        # skip_models must be a list of ModelPhantom (or empty/None)
        if skip_models is None:
            skip_models = []
        if not isinstance(skip_models, list):
            raise TypeError("skip_models must be a list of ModelPhantom objects.")
        for s in skip_models:
            if not hasattr(s, "keys"):
                raise TypeError("skip_models must contain only ModelPhantom objects.")

        skip_keys = {p.keys() for p in skip_models}

        kept_models = []
        skipped_models = []

        for phantom in self._phantoms:
            base_name, dataset_percentage, structure = phantom.keys()
            key = (base_name, dataset_percentage, structure)

            # --- skip requested models ---
            if key in skip_keys:
                skipped_models.append(phantom.string(verbose=True))
                continue

            # init dict nesting
            if self._field_prediction_dict is None:
                self._field_prediction_dict = {}
            level1 = self._field_prediction_dict.setdefault(base_name, {})
            level2 = level1.setdefault(dataset_percentage, {})

            # keep existing predictions if requested
            if keep_existing and structure in level2:
                kept_models.append(phantom.string(verbose=True))
                continue

            # get model (if missing, we cannot compute new preds)
            try:
                model = self._models[base_name][dataset_percentage][structure]
            except KeyError:
                # If preds already exist (even though keep_existing=False), don't delete them—just warn.
                if structure in level2:
                    warnings.warn(
                        f"[compute_field_predictions] No model object for {phantom.string(verbose=True)}; "
                        f"cannot recompute, keeping existing field preds.",
                        RuntimeWarning,
                    )
                else:
                    warnings.warn(
                        f"[compute_field_predictions] No model object for {phantom.string(verbose=True)} "
                        f"and no existing field preds to keep; skipping.",
                        RuntimeWarning,
                    )
                continue

            # choose dataset (fallback to 100% if pct missing; raise if 100% missing)
            pos_arr, cur_arr, _field_arr, used_pct = self._select_dataset_for_phantom(
                phantom, purpose="compute_field_predictions"
            )

            tqdm_msg = (
                f"Computing field predictions for model '{phantom.string(verbose=True)}' "
                f"(dataset {used_pct}%)..."
            )

            preds = []
            for pos, curr in tqdm(
                zip(pos_arr, cur_arr),
                total=len(pos_arr),
                desc=tqdm_msg,
                leave=True,
            ):
                preds.append(model.get_field(pos, curr))

            level2[structure] = np.asarray(preds)

        if skipped_models:
            print("Skipped field predictions for models: " + ", ".join(skipped_models))

        if keep_existing and kept_models:
            print(
                "Kept existing field predictions for models: "
                + ", ".join(kept_models)
                + ". Call keep_existing=False to recompute."
            )

    def compute_gradient_predictions(self, keep_existing=True, skip_models=None):
        if not self._positions or not self._currents:
            raise ValueError("Positions and currents must be loaded before computing predictions.")

        # skip_models must be a list of ModelPhantom (or empty/None)
        if skip_models is None:
            skip_models = []
        if not isinstance(skip_models, list):
            raise TypeError("skip_models must be a list of ModelPhantom objects.")
        for s in skip_models:
            if not hasattr(s, "keys"):
                raise TypeError("skip_models must contain only ModelPhantom objects.")

        # compare via canonical key triples
        skip_keys = {p.keys() for p in skip_models}

        kept_models = []
        skipped_models = []

        for phantom in self._phantoms:
            base_name, dataset_percentage, structure = phantom.keys()
            key = (base_name, dataset_percentage, structure)

            # --- skip requested models ---
            if key in skip_keys:
                skipped_models.append(phantom.string(verbose=True))
                continue

            # init dict nesting
            if self._gradient_prediction_dict is None:
                self._gradient_prediction_dict = {}
            level1 = self._gradient_prediction_dict.setdefault(base_name, {})
            level2 = level1.setdefault(dataset_percentage, {})

            # keep existing predictions if requested
            if keep_existing and structure in level2:
                kept_models.append(phantom.string(verbose=True))
                continue

            # get model (if missing, we cannot compute new preds)
            try:
                model = self._models[base_name][dataset_percentage][structure]
            except KeyError:
                warnings.warn(
                    f"[compute_gradient_predictions] No model object for {phantom.string(verbose=True)} "
                    f"and no existing gradient preds to keep; skipping.",
                    RuntimeWarning,
                )
                continue

            # choose dataset (fallback to 100% if pct missing; raise if 100% missing)
            pos_arr, cur_arr, _field_arr, used_pct = self._select_dataset_for_phantom(
                phantom, purpose="compute_gradient_predictions"
            )

            tqdm_msg = (
                f"Computing gradient predictions for model '{phantom.string(verbose=True)}' "
                f"(dataset {used_pct}%)..."
            )

            preds = []
            for pos, curr in tqdm(
                zip(pos_arr, cur_arr),
                total=len(pos_arr),
                desc=tqdm_msg,
                leave=True,
            ):
                preds.append(model.get_grad9(pos, curr))

            level2[structure] = np.asarray(preds)

        if skipped_models:
            print("Skipped gradient predictions for models: " + ", ".join(skipped_models))

        if keep_existing and kept_models:
            print(
                "Kept existing gradient predictions for models: "
                + ", ".join(kept_models)
                + ". Call keep_existing=False to recompute."
            )

    def apply_field_metric(self, metric_func, keep_existing=True):
        if not self._fields:
            raise ValueError("No datasets loaded in self._fields.")
        if not self._field_prediction_dict:
            raise ValueError("Field predictions must be computed before applying metrics.")
        if self._field_metrics_dict is None:
            self._field_metrics_dict = {}

        kept = []

        for phantom in self._phantoms:
            base_name, dataset_percentage, structure = phantom.keys()

            try:
                model_preds = self._field_prediction_dict[base_name][dataset_percentage][structure]
            except Exception:
                warnings.warn(
                    f"[apply_field_metric] Missing preds for {phantom.string(verbose=True)}; skipping.",
                    RuntimeWarning
                )
                continue

            # choose true fields consistent with how predictions were computed
            _pos_arr, _cur_arr, true_fields, used_pct = self._select_dataset_for_phantom(
                phantom, purpose="apply_field_metric"
            )

            # sanity: lengths should match
            if true_fields is None:
                raise ValueError("Fields are missing for metric computation.")
            if len(true_fields) != len(model_preds):
                warnings.warn(
                    f"[apply_field_metric] Length mismatch for {phantom.string(verbose=True)} "
                    f"(dataset {used_pct}%: true={len(true_fields)}, preds={len(model_preds)}). Skipping.",
                    RuntimeWarning
                )
                continue

            level1 = self._field_metrics_dict.setdefault(base_name, {})
            level2 = level1.setdefault(dataset_percentage, {})
            level3 = level2.setdefault(structure, {})

            out = metric_func(true_fields, model_preds)
            if not isinstance(out, dict):
                raise TypeError(
                    f"metric_func must return dict[str, Any], got {type(out)} for model {phantom.string(verbose=True)}"
                )

            for k, v in out.items():
                if keep_existing and k in level3:
                    kept.append((phantom.string(verbose=True), k))
                    continue
                level3[k] = v

        if keep_existing and kept:
            by_model = {}
            for m, k in kept:
                by_model.setdefault(m, []).append(k)
            msg = "; ".join(f"{m}: {', '.join(ks)}" for m, ks in by_model.items())
            print("Kept existing field metrics (not overwritten): " + msg)

    def apply_gradient_metric(self, metric_func, keep_existing=True):
        if self._gradient_prediction_dict is None:
            raise ValueError("Gradient predictions must be computed before applying metrics.")

        if self._gradient_metrics_dict is None:
            self._gradient_metrics_dict = {}

        kept = []

        for phantom in self._phantoms:
            base_name, dataset_percentage, structure = phantom.keys()

            try:
                model_preds = self._gradient_prediction_dict[base_name][dataset_percentage][structure]
            except Exception:
                warnings.warn(f"[apply_gradient_metric] Missing preds for {phantom.string(verbose=True)}; skipping.", RuntimeWarning)
                continue

            reference_grads, used_pct = self._select_gradient_reference_for_phantom(
                phantom, purpose="apply_gradient_metric"
            )

            if reference_grads is not None and len(reference_grads) != len(model_preds):
                warnings.warn(
                    f"[apply_gradient_metric] Length mismatch for {phantom.string(verbose=True)} "
                    f"(dataset {used_pct}% refs: true={len(reference_grads)}, preds={len(model_preds)}). Skipping.",
                    RuntimeWarning
                )
                continue

            level1 = self._gradient_metrics_dict.setdefault(base_name, {})
            level2 = level1.setdefault(dataset_percentage, {})
            level3 = level2.setdefault(structure, {})

            out = metric_func(reference_grads, model_preds)
            if not isinstance(out, dict):
                raise TypeError(
                    f"metric_func must return dict[str, Any], got {type(out)} for model {phantom.string(verbose=True)}"
                )

            for k, v in out.items():
                if keep_existing and k in level3:
                    kept.append((phantom.string(verbose=True), k))
                    continue
                level3[k] = v

        if keep_existing and kept:
            by_model = {}
            for m, k in kept:
                by_model.setdefault(m, []).append(k)
            msg = "; ".join(f"{m}: {', '.join(ks)}" for m, ks in by_model.items())
            print("Kept existing gradient metrics (not overwritten): " + msg)

    def get_full_dict(self):
        full_dict = {
            "labels": {
                "positions": self._positions,  # dict: {pct: np.ndarray}
                "currents":  self._currents,   # dict: {pct: np.ndarray}
                "fields":    self._fields,     # dict: {pct: np.ndarray}
            },
            "models": {},
        }

        if self._gradient_reference_dict is not None:
            full_dict["gradient_references"] = self._gradient_reference_dict

        for phantom in self._phantoms:
            base_name, dataset_percentage, structure = phantom.keys()

            slot = (
                full_dict["models"]
                .setdefault(base_name, {})
                .setdefault(dataset_percentage, {})
                .setdefault(structure, {})
            )

            # Field preds
            try:
                slot["fields"] = self._field_prediction_dict[base_name][dataset_percentage][structure]
            except Exception:
                print(
                    f"[get_full_dict] Missing field preds for {phantom.string(verbose=True)}; skipping. "
                    "Call compute_field_predictions() to compute them."
                )
                continue

            # Gradient preds
            try:
                slot["gradients"] = self._gradient_prediction_dict[base_name][dataset_percentage][structure]
            except Exception:
                print(
                    f"[get_full_dict] Missing gradient preds for {phantom.string(verbose=True)}; skipping. "
                    "Call compute_gradient_predictions() to compute them."
                )
                continue

        return full_dict
    
    def store(self, filepath):
        full_dict = self.get_full_dict()

        parent = os.path.dirname(os.path.abspath(filepath))
        if parent:
            os.makedirs(parent, exist_ok=True)
            
        with open(filepath, 'wb') as f:
            pickle.dump(full_dict, f)

    @classmethod
    def load_from(cls, filepath):
        with open(filepath, "rb") as f:
            full_dict = pickle.load(f)

        pkg = cls()


        labels = full_dict.get("labels", {})
        pos = labels.get("positions", None)
        cur = labels.get("currents", None)
        fld = labels.get("fields", None)

        if not (isinstance(pos, dict) and isinstance(cur, dict) and isinstance(fld, dict)):
            raise ValueError(
                "[load_from] New-format only: labels.positions/currents/fields must all be dicts "
                "keyed by dataset_percentage (ints)."
            )

        def _normalize_pct_dict(d, name):
            out = {}
            for k, v in d.items():
                if isinstance(k, (int, np.integer)):
                    out[int(k)] = v
                else:
                    raise TypeError(
                        f"[load_from] New-format only: labels['{name}'] has non-integer key {k!r} "
                        f"of type {type(k)}. Keys must be ints (dataset_percentage)."
                    )
            return out

        pkg._positions = _normalize_pct_dict(pos, "positions")
        pkg._currents  = _normalize_pct_dict(cur, "currents")
        pkg._fields    = _normalize_pct_dict(fld, "fields")

        pkg._gradient_reference_dict = full_dict.get("gradient_references", None)

        # We do NOT load model objects
        pkg._models = {}

        # ----------------------------
        # Strict new-format models
        # models[base_name][dataset_percentage][structure] = payload_dict
        # ----------------------------
        models_tree = full_dict.get("models", {})
        if models_tree is None:
            models_tree = {}

        if not isinstance(models_tree, dict):
            raise TypeError("[load_from] New-format only: top-level 'models' must be a dict.")

        if not models_tree:
            pkg._phantoms = []
            pkg._field_prediction_dict = None
            pkg._gradient_prediction_dict = None
            pkg._field_metrics_dict = None
            pkg._gradient_metrics_dict = None
            return pkg

        pkg._phantoms = []
        pkg._field_prediction_dict = {}
        pkg._gradient_prediction_dict = {}
        pkg._field_metrics_dict = {}
        pkg._gradient_metrics_dict = {}

        for base_name, pct_dict in models_tree.items():
            if not isinstance(pct_dict, dict):
                raise TypeError(
                    f"[load_from] New-format only: models['{base_name}'] must be a dict keyed by dataset_percentage."
                )

            for pct_key, struct_dict in pct_dict.items():
                if not isinstance(pct_key, (int, np.integer)):
                    raise TypeError(
                        f"[load_from] New-format only: models['{base_name}'] has non-integer dataset key "
                        f"{pct_key!r} of type {type(pct_key)}. Keys must be ints (dataset_percentage)."
                    )
                dataset_percentage = int(pct_key)

                if not isinstance(struct_dict, dict):
                    raise TypeError(
                        f"[load_from] New-format only: models['{base_name}'][{dataset_percentage}] must be a dict."
                    )

                for structure, payload in struct_dict.items():
                    if not isinstance(payload, dict):
                        raise TypeError(
                            f"[load_from] New-format only: models['{base_name}'][{dataset_percentage}][{structure!r}] "
                            f"must be a dict payload."
                        )

                    # Normalize structure if it came back as list from pickle/json
                    if isinstance(structure, list):
                        structure = tuple(structure)

                    phantom = ModelPhantom(
                        name=base_name,
                        dataset_percentage=dataset_percentage,
                        structure=structure
                    )
                    pkg._phantoms.append(phantom)

                    if "fields" in payload:
                        pkg._field_prediction_dict \
                            .setdefault(base_name, {}) \
                            .setdefault(dataset_percentage, {})[structure] = payload["fields"]

                    if "gradients" in payload:
                        pkg._gradient_prediction_dict \
                            .setdefault(base_name, {}) \
                            .setdefault(dataset_percentage, {})[structure] = payload["gradients"]

                    if "field_metrics" in payload:
                        pkg._field_metrics_dict \
                            .setdefault(base_name, {}) \
                            .setdefault(dataset_percentage, {})[structure] = payload["field_metrics"]

                    if "gradient_metrics" in payload:
                        pkg._gradient_metrics_dict \
                            .setdefault(base_name, {}) \
                            .setdefault(dataset_percentage, {})[structure] = payload["gradient_metrics"]

        # Cosmetic: empty dicts -> None
        if not pkg._field_prediction_dict:
            pkg._field_prediction_dict = None
        if not pkg._gradient_prediction_dict:
            pkg._gradient_prediction_dict = None
        if not pkg._field_metrics_dict:
            pkg._field_metrics_dict = None
        if not pkg._gradient_metrics_dict:
            pkg._gradient_metrics_dict = None

        return pkg
    

    def merge_dict(self, other_dict, *, force=False, warn_on_conflict=True):
        other_labels = other_dict.get("labels", {})
        other_positions = other_labels.get("positions", None)
        other_currents  = other_labels.get("currents", None)
        other_fields    = other_labels.get("fields", None)

        if other_positions is None or other_currents is None or other_fields is None:
            raise ValueError("other_dict must contain labels: positions, currents, fields")

        # Strict new-format: labels must be dicts
        if not (isinstance(other_positions, dict) and isinstance(other_currents, dict) and isinstance(other_fields, dict)):
            raise TypeError("other_dict labels must be dicts keyed by dataset_percentage (new format only).")

        if not (isinstance(self._positions, dict) and isinstance(self._currents, dict) and isinstance(self._fields, dict)):
            raise TypeError("This package labels must be dicts keyed by dataset_percentage (new format only).")

        def _normalize_pct_keys(d, name):
            out = {}
            for k, v in d.items():
                if isinstance(k, (int, np.integer)):
                    out[int(k)] = v
                else:
                    raise TypeError(
                        f"[merge_dict] other_dict labels['{name}'] has non-integer key {k!r} of type {type(k)}. "
                        "Keys must be ints (dataset_percentage)."
                    )
            return out

        other_positions = _normalize_pct_keys(other_positions, "positions")
        other_currents  = _normalize_pct_keys(other_currents,  "currents")
        other_fields    = _normalize_pct_keys(other_fields,    "fields")

        # ---- merge/validate labels by pct ----
        other_pcts = set(other_positions.keys()) | set(other_currents.keys()) | set(other_fields.keys())
        for pct in other_pcts:
            if pct not in other_positions or pct not in other_currents or pct not in other_fields:
                raise ValueError(f"other_dict has partial labels for dataset_percentage={pct} (must have positions/currents/fields).")

            if pct in self._positions or pct in self._currents or pct in self._fields:
                # if any exist, require all exist
                if pct not in self._positions or pct not in self._currents or pct not in self._fields:
                    raise ValueError(f"This package has partial labels for dataset_percentage={pct}; cannot merge safely.")

                # validate equality
                if not np.array_equal(self._positions[pct], other_positions[pct]):
                    raise ValueError(f"Positions for dataset_percentage={pct} do not match.")
                if not np.array_equal(self._currents[pct], other_currents[pct]):
                    raise ValueError(f"Currents for dataset_percentage={pct} do not match.")
                if not np.array_equal(self._fields[pct], other_fields[pct]):
                    raise ValueError(f"Fields for dataset_percentage={pct} do not match.")
            else:
                self._positions[pct] = other_positions[pct]
                self._currents[pct]  = other_currents[pct]
                self._fields[pct]    = other_fields[pct]

        # ---- gradient references (if present, must match) ----
        if "gradient_references" in other_dict:
            other_ref = other_dict["gradient_references"]
            if self._gradient_reference_dict is None:
                self._gradient_reference_dict = other_ref
            else:
                if (
                    not np.array_equal(self._gradient_reference_dict.get("positions", None), other_ref.get("positions", None))
                    or not np.array_equal(self._gradient_reference_dict.get("currents", None),  other_ref.get("currents", None))
                    or not np.array_equal(self._gradient_reference_dict.get("fields", None),    other_ref.get("fields", None))
                ):
                    raise ValueError("gradient_references in other_dict do not match existing gradient_references.")

        # ---- ensure top dicts ----
        if self._field_prediction_dict is None:
            self._field_prediction_dict = {}
        if self._gradient_prediction_dict is None:
            self._gradient_prediction_dict = {}
        if self._field_metrics_dict is None:
            self._field_metrics_dict = {}
        if self._gradient_metrics_dict is None:
            self._gradient_metrics_dict = {}

        def _phantom_exists(base_name, dataset_percentage, structure):
            for p in self._phantoms:
                b, dp, s = p.keys()
                if b == base_name and dp == dataset_percentage and s == structure:
                    return True
            return False

        # ---- merge model payloads (strict new-format only!!!) ----
        models_tree = other_dict.get("models", {})
        if models_tree is None:
            models_tree = {}

        if not isinstance(models_tree, dict):
            raise TypeError("[merge_dict] other_dict['models'] must be a dict (new format only).")

        for base_name, pct_dict in models_tree.items():
            if not isinstance(pct_dict, dict):
                raise TypeError(f"[merge_dict] other_dict['models']['{base_name}'] must be a dict keyed by dataset_percentage.")

            for pct_key, struct_dict in pct_dict.items():
                if not isinstance(pct_key, (int, np.integer)):
                    raise TypeError(
                        f"[merge_dict] other_dict['models']['{base_name}'] has non-integer dataset key {pct_key!r} "
                        f"of type {type(pct_key)}. Keys must be ints (dataset_percentage)."
                    )
                dataset_percentage = int(pct_key)

                if not isinstance(struct_dict, dict):
                    raise TypeError(
                        f"[merge_dict] other_dict['models']['{base_name}'][{dataset_percentage}] must be a dict."
                    )

                for structure, payload in struct_dict.items():
                    if not isinstance(payload, dict):
                        raise TypeError(
                            f"[merge_dict] payload at models['{base_name}'][{dataset_percentage}][{structure!r}] must be a dict."
                        )

                    # normalize structure if it came back as list from pickle/json
                    if isinstance(structure, list):
                        structure = tuple(structure)

                    if not _phantom_exists(base_name, dataset_percentage, structure):
                        self._phantoms.append(
                            ModelPhantom(name=base_name, dataset_percentage=dataset_percentage, structure=structure)
                        )

                    slot_desc = f"{base_name}/{dataset_percentage}/{structure}"

                    # preds
                    if "fields" in payload:
                        d = self._field_prediction_dict.setdefault(base_name, {}).setdefault(dataset_percentage, {})
                        if (structure in d) and (not force):
                            if warn_on_conflict:
                                warnings.warn(
                                    f"[merge_dict] Keeping existing fields for {slot_desc} (force=False).",
                                    RuntimeWarning
                                )
                        else:
                            d[structure] = payload["fields"]

                    if "gradients" in payload:
                        d = self._gradient_prediction_dict.setdefault(base_name, {}).setdefault(dataset_percentage, {})
                        if (structure in d) and (not force):
                            if warn_on_conflict:
                                warnings.warn(
                                    f"[merge_dict] Keeping existing gradients for {slot_desc} (force=False).",
                                    RuntimeWarning
                                )
                        else:
                            d[structure] = payload["gradients"]

                    # metrics: merge by metric key
                    if "field_metrics" in payload:
                        dst = self._field_metrics_dict.setdefault(base_name, {}).setdefault(dataset_percentage, {}).setdefault(structure, {})
                        src = payload["field_metrics"]
                        if isinstance(src, dict):
                            for mk, mv in src.items():
                                if mk in dst and (not force):
                                    if warn_on_conflict:
                                        warnings.warn(
                                            f"[merge_dict] Keeping existing field_metric '{mk}' for {slot_desc} (force=False).",
                                            RuntimeWarning
                                        )
                                    continue
                                dst[mk] = mv
                        else:
                            # strict: keep prior behavior, but still allow overwrite if force
                            if force or ("field_metrics" not in dst):
                                dst["field_metrics"] = src

                    if "gradient_metrics" in payload:
                        dst = self._gradient_metrics_dict.setdefault(base_name, {}).setdefault(dataset_percentage, {}).setdefault(structure, {})
                        src = payload["gradient_metrics"]
                        if isinstance(src, dict):
                            for mk, mv in src.items():
                                if mk in dst and (not force):
                                    if warn_on_conflict:
                                        warnings.warn(
                                            f"[merge_dict] Keeping existing gradient_metric '{mk}' for {slot_desc} (force=False).",
                                            RuntimeWarning
                                        )
                                    continue
                                dst[mk] = mv
                        else:
                            if force or ("gradient_metrics" not in dst):
                                dst["gradient_metrics"] = src

        # cosmetic
        if self._field_prediction_dict == {}:
            self._field_prediction_dict = None
        if self._gradient_prediction_dict == {}:
            self._gradient_prediction_dict = None
        if self._field_metrics_dict == {}:
            self._field_metrics_dict = None
        if self._gradient_metrics_dict == {}:
            self._gradient_metrics_dict = None

    def _select_dataset_for_phantom(self, phantom, *, purpose="compute"):
        """
        Decide which (positions, currents, fields) arrays to use for this phantom.

        Rules:
        - Prefer the phantom's dataset_percentage (pct).
        - If pct is missing, fall back to 100.
        - If 100 is also missing, raise.

        Returns
        -------
        positions, currents, fields : np.ndarray
        used_pct : int
        """
        pct = int(getattr(phantom, "dataset_percentage", 100))

        def _has(p):
            return (
                isinstance(self._positions, dict) and p in self._positions
                and isinstance(self._currents, dict) and p in self._currents
                and isinstance(self._fields, dict) and p in self._fields
            )

        if _has(pct):
            return self._positions[pct], self._currents[pct], self._fields[pct], pct

        # fallback to full
        if _has(100):
            warnings.warn(
                f"[{purpose}] Dataset {pct}% missing for model '{phantom.string(verbose=True)}'; "
                "falling back to 100%.",
                RuntimeWarning,
            )
            return self._positions[100], self._currents[100], self._fields[100], 100

        raise ValueError(
            f"[{purpose}] Dataset {pct}% missing for model '{phantom.string(verbose=True)}', "
            "and full dataset (100%) is also not loaded."
        )
    
    def _select_gradient_reference_for_phantom(self, phantom, *, purpose="apply_gradient_metric", decimals=None):
        """
        Returns (ref_grads_subset_or_None, used_pct)

        Rules:
        - Gradient references are ALWAYS loaded for the FULL dataset (ref dict contains full positions/currents/fields).
        - For a phantom at dataset_percentage=pct:
            - select that dataset's (pos, curr, fields) via _select_dataset_for_phantom (includes fallback-to-100 rule)
            - find matching rows in the FULL reference labels (by pos/curr/field)
            - return ref_grads_full indexed to match the dataset order
        - If refs missing / chosen params missing -> warn and return (None, used_pct)
        """
        # choose which dataset we are evaluating on (pct or fallback to 100)
        pos_arr, cur_arr, fld_arr, used_pct = self._select_dataset_for_phantom(
            phantom, purpose=purpose
        )

        full_ref = getattr(self, "_gradient_reference_dict", None)
        if full_ref is None:
            return None, used_pct

        # validate presence of label arrays in reference dict
        for k in ("positions", "currents", "fields", "gradients"):
            if k not in full_ref:
                warnings.warn(
                    f"[{purpose}] Reference dict missing key '{k}'; passing None for {phantom.string(verbose=True)}.",
                    RuntimeWarning
                )
                return None, used_pct

        ref_pos = np.asarray(full_ref["positions"])
        ref_cur = np.asarray(full_ref["currents"])
        ref_fld = np.asarray(full_ref["fields"])

        grads_tree = full_ref.get("gradients", None)
        if grads_tree is None or not isinstance(grads_tree, dict):
            warnings.warn(
                f"[{purpose}] Reference dict has no usable 'gradients' for {phantom.string(verbose=True)}; passing None.",
                RuntimeWarning
            )
            return None, used_pct

        if self._chosen_order is None or self._chosen_neighbor_count is None:
            warnings.warn(
                f"[{purpose}] chosen_order/neighbor_count not set; passing None refs to metric for {phantom.string(verbose=True)}.",
                RuntimeWarning
            )
            return None, used_pct

        try:
            ref_grads_full = np.asarray(grads_tree[self._chosen_order][self._chosen_neighbor_count])
        except Exception:
            warnings.warn(
                f"[{purpose}] Missing refs for order={self._chosen_order}, nn={self._chosen_neighbor_count} "
                f"for {phantom.string(verbose=True)}; passing None.",
                RuntimeWarning
            )
            return None, used_pct

        # If we're evaluating on full dataset, no subsetting needed
        if used_pct == 100:
            return ref_grads_full, 100

        # --- build row keys and align ---
        ref_keys = _make_row_keys(ref_pos, ref_cur, ref_fld, decimals=decimals)
        data_keys = _make_row_keys(pos_arr, cur_arr, fld_arr, decimals=decimals)

        # map reference key -> index (assumes unique; if not unique, we warn and keep first)
        index_map = {}
        dup_count = 0
        for i, k in enumerate(ref_keys):
            if k in index_map:
                dup_count += 1
                continue
            index_map[k] = i
        if dup_count:
            warnings.warn(
                f"[{purpose}] Reference label keys had {dup_count} duplicates; using first occurrence for matching.",
                RuntimeWarning
            )

        # gather indices in DATA ORDER (so ref_grads matches your predictions order)
        idx = np.empty(len(data_keys), dtype=int)
        missing = 0
        for j, k in enumerate(data_keys):
            i = index_map.get(k, None)
            if i is None:
                missing += 1
                idx[j] = -1
            else:
                idx[j] = i

        if missing:
            raise ValueError(
                f"[{purpose}] Could not match {missing}/{len(data_keys)} rows from dataset {used_pct}% "
                f"to the full reference labels for model '{phantom.string(verbose=True)}'. "
                "Consider setting `decimals` to something like 6-12 if this is floating-point mismatch."
            )

        return ref_grads_full[idx], used_pct
    
        
def _make_row_keys(positions, currents, fields, *, decimals=None):
    """
    Convert per-sample (pos,curr,field) rows into hashable keys.
    Uses a packed byte representation of concatenated floats.

    decimals:
    - None -> exact float bytes
    - int  -> round to that many decimals before hashing (useful for float mismatch)
    """
    p = np.asarray(positions)
    c = np.asarray(currents)
    f = np.asarray(fields)

    if len(p) != len(c) or len(p) != len(f):
        raise ValueError("positions/currents/fields must have same length to build keys.")

    # flatten per-sample and concatenate
    P = p.reshape(len(p), -1)
    C = c.reshape(len(c), -1)
    F = f.reshape(len(f), -1)
    X = np.concatenate([P, C, F], axis=1)

    if decimals is not None:
        X = np.round(X, decimals=decimals)

    X = np.ascontiguousarray(X)
    row_dtype = np.dtype((np.void, X.dtype.itemsize * X.shape[1]))
    return X.view(row_dtype).ravel()

def _pop_nested(d, k1, k2, k3, default=None, *, prune=True):
    """
    Pop d[k1][k2][k3] if it exists.
    If prune=True, remove empty parent dicts after popping.
    """
    if d is None:
        return default

    lvl1 = d.get(k1)
    if not isinstance(lvl1, dict):
        return default

    lvl2 = lvl1.get(k2)
    if not isinstance(lvl2, dict):
        return default

    out = lvl2.pop(k3, default)

    if prune:
        if len(lvl2) == 0:
            lvl1.pop(k2, None)
        if len(lvl1) == 0:
            d.pop(k1, None)

    return out