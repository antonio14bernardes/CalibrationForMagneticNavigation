import os
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .calibration import Calibration


class NNCalibration(Calibration, nn.Module, ABC):
    """
    Abstract base for NN-based calibrations.

    - Inherits Calibration (numpy API) and nn.Module (torch training).
    - Implements save/load exactly like your old BaseActuationNet.
    - Optionally normalizes positions and (de)normalizes fields.
    """

    def __init__(
        self,
        num_coils = 8,
        name: str = "NNCALIBRATION",
        position_normalization_dict: Optional[dict] = None,
        field_normalization_dict: Optional[dict] = None,
    ):
        Calibration.__init__(self, name, num_coils=num_coils)
        nn.Module.__init__(self)


        self._position_normalization_dict = position_normalization_dict
        self._field_normalization_dict = field_normalization_dict

        

        if self._position_normalization_dict is not None and self._position_normalization_dict.keys() != {"mean", "std"}:
            raise ValueError("position_normalization_dict must have keys 'mean' and 'std'")
        if self._field_normalization_dict is not None and self._field_normalization_dict.keys() != {"mean", "std"}:
            raise ValueError("field_normalization_dict must have keys 'mean' and 'std'")

    # ------------------------------------------------------------------
    # Config (to be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_config(self) -> dict:
        """
        Return a dict of init kwargs needed to reconstruct this model.
        Example:
            return {
                "hidden_dims": self.hidden_dims,
                "name": self.name,
                "position_normalization_dict": self.position_normalization_dict_,
                "field_normalization_dict": self.field_normalization_dict_,
            }
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------

    def save(self, folder: str) -> str:
        """
        Save model weights + config to `folder`.
        Filename: {name}_YYYYMMDD_HHMMSS.pt
        Returns the full path.
        """
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}.pt"
        path = os.path.join(folder, filename)

        ckpt = {
            "state_dict": self.state_dict(),
            "config": self.get_config(),
            "name": self.name,
        }
        torch.save(ckpt, path)
        return path

    @classmethod
    def load_from(cls, path, map_location="cpu"):
        """
        Load model from checkpoint created by `.save()` / `.store()`.

        No need to pass hidden_dims, n_coils, etc.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        config = ckpt.get("config", {})
        model = cls(**config)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    # ------------------------------------------------------------------
    # numpy <-> torch
    # ------------------------------------------------------------------

    def _to_tensor(self, x: np.ndarray, device=None, dtype=torch.float32) -> torch.Tensor:
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        return torch.as_tensor(x, dtype=dtype, device=device)

    def _to_numpy(self, t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def setup_normalization(
        self,
        position_normalization_dict: Optional[dict] = None,
        field_normalization_dict: Optional[dict] = None,
    ):
        """
        Setup normalization dicts after initialization.

        position_normalization_dict: dict with keys "mean" and "std", each a (3,) array
        field_normalization_dict:    dict with keys "mean" and "std", each a (3,) array
        """
        self._position_normalization_dict = position_normalization_dict
        self._field_normalization_dict = field_normalization_dict

    @property
    def position_normalization_dict(self) -> Optional[dict]:
        return self._position_normalization_dict
    @property
    def field_normalization_dict(self) -> Optional[dict]:
        return self._field_normalization_dict

    def _normalize_input(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Normalize positions using position_normalization_dict_ if provided.

        pos: (N, 3) tensor
        returns: (N, 3) tensor
        """
        if pos.shape[-1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got {tuple(pos.shape)}")

        if self._position_normalization_dict is None:
            return pos

        mean = torch.as_tensor(
            self._position_normalization_dict["mean"],
            device=pos.device,
            dtype=pos.dtype,
        )
        std = torch.as_tensor(
            self._position_normalization_dict["std"],
            device=pos.device,
            dtype=pos.dtype,
        )

        return (pos - mean) / std

    def _denormalize_output(self, field: torch.Tensor) -> torch.Tensor:
        """
        Denormalize fields using field_normalization_dict_ if provided.

        field: (N, 3) tensor
        returns: (N, 3) tensor
        """
        if field.shape[-1] != 3:
            raise ValueError(f"field must have shape (N, 3), got {tuple(field.shape)}")

        if self._field_normalization_dict is None:
            return field

        mean = torch.as_tensor(
            self._field_normalization_dict["mean"],
            device=field.device,
            dtype=field.dtype,
        )
        std = torch.as_tensor(
            self._field_normalization_dict["std"],
            device=field.device,
            dtype=field.dtype,
        )

        return field * std + mean

    # ------------------------------------------------------------------
    # Forward method for nn.Module (to be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, pos: torch.Tensor, currents: torch.Tensor) -> torch.Tensor:
        """
        Map (pos, currents) -> field (N, 3).
        Inputs are torch Tensors (typically already normalized).
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # --------- CALIBRATION INTERFACE MUST STILL BE IMPLEMENTED ------------
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ----------------------------------------------------------------------