import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev
from typing import Optional, Tuple

from .nn_calibration import NNCalibration


class DirectNet(NNCalibration):
    def __init__(
        self,
        num_coils: int = 8,
        hidden_dims=(150, 100, 50),
        name: str = "DirectNet",
        position_normalization_dict: Optional[dict] = None,
        field_normalization_dict: Optional[dict] = None,
    ):
        super().__init__(
            name=name,
            num_coils=num_coils,
            position_normalization_dict=position_normalization_dict,
            field_normalization_dict=field_normalization_dict,
        )
        self.hidden_dims = tuple(hidden_dims)

        input_dim = 3 + self._num_coils
        output_dim = 3

        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    # ----- config for save/load -----
    def get_config(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "num_coils": self.num_coils,
            "name": self.name,
            "position_normalization_dict": self.position_normalization_dict,
            "field_normalization_dict": self.field_normalization_dict,
        }

    # ----- torch forward -----
    def forward(self, pos: torch.Tensor, currents: Optional[torch.Tensor]) -> torch.Tensor:
        """
        pos:      (batch, 3)          (unnormalized; will be normalized internally)
        currents: (batch, n_coils) or (n_coils,)
        returns:  field (batch, 3)    (denormalized / physical units)
        """

        # Currents actually need to be supplied here
        if currents is None:
            raise ValueError("currents must be provided to DirectNet forward pass.")


        # normalize input positions if dict is provided
        pos = self._normalize_input(pos)

        if currents.dim() == 1:
            batch_size = pos.size(0)
            currents = currents.unsqueeze(0).expand(batch_size, -1)

        x = torch.cat([pos, currents], dim=-1)   # (batch, 3 + n_coils)
        field_norm = self.mlp(x)                 # (batch, 3) in normalized field space

        # denormalize output fields if dict is provided
        field = self._denormalize_output(field_norm)

        return field

    # ----- Calibration (numpy) API -----

    def get_field(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        """
        pos:      (3,)          (unnormalized)
        currents: (n_coils,)    (unnormalized or as expected by the model)
        returns:  (3,)          (denormalized / physical units)
        """
        self.eval()
        device = next(self.parameters()).device

        # Convert to tensors
        pos_t = self._to_tensor(pos, device=device)
        cur_t = self._to_tensor(currents, device=device)

        # Add batch dimension
        pos_t = pos_t.view(1, -1)
        cur_t = cur_t.view(1, -1)

        with torch.no_grad():
            field = self.forward(pos_t, cur_t)

        # Back to numpy, drop batch dim -> (3,)
        field_np = self._to_numpy(field)[0] * self._multiplier
        return field_np
    
    def get_grad5(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        _, grad5 = self.get_field_and_grad5(pos, currents)
        return grad5
    
    def get_grad9(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        _, grad9 = self.get_field_and_grad9(pos, currents)
        return grad9
    
    def get_field_and_grad5(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
        field, grad9 = self.get_field_and_grad9(pos, currents)
        dbx_dxyz = grad9[0, :]
        dby_dyz  = grad9[1, 1:]
        grad5 = np.concatenate([dbx_dxyz, dby_dyz])
        return field, grad5
    
    def get_field_and_grad9_old(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the field and its Jacobian w.r.t. position for DirectNet.

        pos:      (3,)          (unnormalized)
        currents: (n_coils,)    (unnormalized or as expected by the model)

        Returns:
            field: (3,)          field in physical units at (pos, currents)
            grad9: (3, 3)        Jacobian d(field) / d(pos)
                                grad9[i, j] = d field[i] / d pos[j]
        """
        self.eval()
        device = next(self.parameters()).device

        # to torch, add batch dim
        pos_t = self._to_tensor(pos, device=device).view(1, -1)        # (1, 3)
        cur_t = self._to_tensor(currents, device=device).view(1, -1)   # (1, n_coils)

        # we need gradients w.r.t. pos
        pos_t.requires_grad_(True)

        # DO NOT wrap in torch.no_grad() here!
        field = self.forward(pos_t, cur_t)  # (1, 3)

        # --- compute Jacobian d(field)/d(pos) using autograd.grad ---
        jac_rows = []
        for i in range(3):
            # grad_outputs selects which output component to backprop from.
            # This effectively computes the gradient of field[0, i] w.r.t. pos_t.
            grad_outputs = torch.zeros_like(field)  # (1, 3)
            grad_outputs[0, i] = 1.0

            (grad_pos,) = torch.autograd.grad(
                outputs=field,
                inputs=pos_t,
                grad_outputs=grad_outputs,
                retain_graph=True,   # keep graph for next i
                create_graph=False,  # no higher-order grads needed
            )
            # grad_pos: (1, 3) -> (3,)
            jac_rows.append(grad_pos[0])

        jacobian = torch.stack(jac_rows, dim=0)  # (3, 3)

        # --- convert to numpy ---
        field_np = self._to_numpy(field)[0]      # (3,)
        grad9_np = self._to_numpy(jacobian)      # (3, 3)

        return field_np, grad9_np
    
    def get_field_and_grad9(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the field and its Jacobian w.r.t. position for DirectNet.

        pos:      (3,)          (unnormalized)
        currents: (n_coils,)    (unnormalized or as expected by the model)

        Returns:
            field: (3,)          field in physical units at (pos, currents)
            grad9: (3, 3)        Jacobian d(field) / d(pos)
                                grad9[i, j] = d field[i] / d pos[j]
        """
        self.eval()
        device = next(self.parameters()).device

        # to torch, add batch dim
        pos_t = self._to_tensor(pos, device=device).view(1, -1)        # (1, 3)
        cur_t = self._to_tensor(currents, device=device).view(1, -1)   # (1, n_coils)

        # ----- define function field(pos_vec) with currents fixed -----
        def field_wrt_pos(pos_vec: torch.Tensor) -> torch.Tensor:
            """
            pos_vec: (3,)
            returns: (3,) field at that position, with fixed currents
            """
            pos_vec = pos_vec.view(1, -1)          # (1, 3)
            field = self.forward(pos_vec, cur_t)   # (1, 3)
            return field.view(-1)                  # (3,)

        # pos_flat: (3,)
        pos_flat = pos_t.view(-1)

        # ---- Jacobian d(field)/d(pos) via jacrev ----
        # jacobian: (3, 3) where jacobian[i, j] = d field[i] / d pos[j]
        jacobian = jacrev(field_wrt_pos)(pos_flat)

        # ---- field value itself (no grad needed) ----
        with torch.no_grad():
            field = self.forward(pos_t, cur_t)     # (1, 3)

        # --- convert to numpy ---
        field_np = self._to_numpy(field)[0] * self._multiplier       # (3,)
        grad9_np = self._to_numpy(jacobian) * self._multiplier       # (3, 3)

        return field_np, grad9_np

    def currents_field_jacobian(self, pos: np.ndarray) -> np.ndarray:
        """
        For DirectNet the mapping is generally nonlinear in currents.
        Without specifying an operating point in currents, this Jacobian
        is not well-defined - so we raise an explicit error.
        """
        raise NotImplementedError(
            "currents_field_jacobian is not yet defined for DirectNet "
            "without specifying an operating point in currents."
        )

    def currents_grad5_jacobian(self, pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError("currents_grad5_jacobian is not yet defined for DirectNet.")

    def currents_full_jacobian(self, pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError("currents_full_jacobian is not yet defined for DirectNet.")

    def get_currents(self, pos: np.ndarray, target_field: np.ndarray = None, target_grad5: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError("get_currents is not yet defined for DirectNet.")