import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev 
from typing import Optional, Tuple

from .nn_calibration import NNCalibration


class LinearNet(NNCalibration):
    def __init__(
        self,
        hidden_dims=(150, 100, 50),
        name: str = "LinearNet",
        num_coils: int = 8,
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

        input_dim = 3
        output_dim = 3 * self.num_coils  # 24

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

    # ----- torch forward (+ normalization) -----

    def forward(self, pos: torch.Tensor, currents: Optional[torch.Tensor] = None):
        """
        pos:      (batch, 3)  (unnormalized; will be normalized internally)
        currents: (batch, 8) or (8,) or None

        If currents is None: returns J_phys  (batch, 3, 8)
        else: returns (field_phys, J_phys)
        """
        # normalize positions if dict is set
        pos = self._normalize_input(pos)

        flat = self.mlp(pos)                      # (batch, 24)
        J_norm = flat.view(-1, 3, self.num_coils)   # (batch, 3, 8)

        # convert J to physical units if field normalization is used
        if self.field_normalization_dict is None:
            J_phys = J_norm
        else:
            std = torch.as_tensor(
                self.field_normalization_dict["std"],
                device=J_norm.device,
                dtype=J_norm.dtype,
            )  # (3,)
            J_phys = J_norm * std.view(1, 3, 1)   # (batch, 3, 8)
            J_phys *= self._multiplier

        if currents is None:
            return J_phys

        if currents.dim() == 1:
            currents = currents.unsqueeze(0).expand(J_phys.size(0), -1)  # (batch, 8)

        # field in physical units
        field = torch.einsum("bij,bj->bi", J_phys, currents)             # (batch, 3)

        return field, J_phys

    # ----- Calibration (numpy) API -----

    def get_field(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        """
        pos:      (3,)          (unnormalized)
        currents: (num_coils,)  == (8,)
        returns:  (3,)          (physical units)
        """
        self.eval()
        device = next(self.parameters()).device

        # to torch
        pos_t = self._to_tensor(pos, device=device)          # (3,)
        cur_t = self._to_tensor(currents, device=device)     # (8,)

        # add batch dimension
        pos_t = pos_t.view(1, -1)        # (1, 3)
        cur_t = cur_t.view(1, -1)        # (1, 8)

        with torch.no_grad():
            field, _ = self.forward(pos_t, cur_t)  # (1, 3)

        # remove batch dimension -> (3,)
        field_np = self._to_numpy(field).reshape(-1)
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
        pos:      (3,)          (unnormalized)
        currents: (num_coils,)  == (8,)
        returns:
            - field: (3,)        (physical units)
            - grad9: (3,3)        flattened Jacobian d(field)/d(pos)
                                order: grad9[3*i + j] = d field[i] / d pos[j]
        """
        self.eval()
        device = next(self.parameters()).device

        # to torch
        pos_t = self._to_tensor(pos, device=device)      # (3,)
        cur_t = self._to_tensor(currents, device=device) # (8,)

        # add batch dimension
        pos_t = pos_t.view(1, -1)    # (1, 3)
        cur_t = cur_t.view(1, -1)    # (1, 8)

        # we need gradients w.r.t. pos
        pos_t.requires_grad_(True)

        # DO NOT wrap in torch.no_grad() here!
        field, _ = self.forward(pos_t, cur_t)  # field: (1, 3)

        # --- compute Jacobian d(field)/d(pos) using autograd.grad ---
        jac_rows = []
        for i in range(3):
            # grad_outputs selects which output component to backprop from. THis then gives the vector product vT J where v is grad_outputs
            grad_outputs = torch.zeros_like(field)  # (1, 3)
            grad_outputs[0, i] = 1.0               # d field[i] / d pos

            (grad_pos,) = torch.autograd.grad(
                outputs=field,
                inputs=pos_t,
                grad_outputs=grad_outputs,
                retain_graph=True,   # keep graph for next i
                create_graph=False,  # we don't need higher-order grads
            )
            # grad_pos: (1, 3) -> drop batch -> (3,)
            jac_rows.append(grad_pos[0])

        # stack rows -> (3, 3)
        jacobian = torch.stack(jac_rows, dim=0)  # row i = d field[i,:] / d pos[:]

        # --- convert to numpy ---

        field_np = self._to_numpy(field)[0]                  # (3,)
        grad9_np = self._to_numpy(jacobian)      # (9,)

        return field_np, grad9_np
    
    def get_field_and_grad9(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        pos:      (3,)          (unnormalized)
        currents: (num_coils,)  == (8,)
        returns:
            - field: (3,)        (physical units)
            - grad9: (3,3)       Jacobian d(field)/d(pos)
                                order: grad9[i, j] = d field[i] / d pos[j]
        """
        self.eval()
        device = next(self.parameters()).device

        # to torch
        pos_t = self._to_tensor(pos, device=device)      # (3,)
        cur_t = self._to_tensor(currents, device=device) # (8,)

        # add batch dimension
        pos_t = pos_t.view(1, -1)    # (1, 3)
        cur_t = cur_t.view(1, -1)    # (1, 8)

        # ----- define function field(pos_vec) with currents fixed -----
        def field_wrt_pos(pos_vec: torch.Tensor) -> torch.Tensor:
            """
            pos_vec: (3,)
            returns: (3,) field at that position, with fixed currents
            """
            pos_vec = pos_vec.view(1, -1)             # (1, 3)
            field, _ = self.forward(pos_vec, cur_t)   # (1, 3)
            return field.view(-1)                     # (3,)

        # pos_flat: (3,)
        pos_flat = pos_t.view(-1)

        # ---- Jacobian d(field)/d(pos) via jacrev ----
        # jacobian: (3, 3) where jacobian[i, j] = d field[i] / d pos[j]
        jacobian = jacrev(field_wrt_pos)(pos_flat)

        # ---- field value itself (no grad needed) ----
        with torch.no_grad():
            field, _ = self.forward(pos_t, cur_t)     # (1, 3)

        # --- convert to numpy ---
        field_np = self._to_numpy(field)[0]          # (3,)
        grad9_np = self._to_numpy(jacobian)          # (3, 3)

        return field_np, grad9_np
    

    def currents_field_jacobian(self, pos: np.ndarray) -> np.ndarray:
        """
        Because the model is linear in currents, d field / d currents = J_phys(pos).

        pos:     (3,)       (unnormalized; normalization applied internally)
        returns: (3, num_coils)     in physical units
        """
        self.eval()
        device = next(self.parameters()).device

        pos_t = self._to_tensor(pos, device=device)  # (3,)
        pos_t = pos_t.view(1, -1)                    # (1, 3)

        with torch.no_grad():
            J_phys = self.forward(pos_t, currents=None)  # (1, 3, 8)

        J_np = self._to_numpy(J_phys)[0]   # (3, 8)
        return J_np

    def currents_grad5_jacobian(self, pos: np.ndarray) -> np.ndarray:
        J_full = self.currents_full_jacobian(pos)  # (8, num_coils)
        J_grad5 = J_full[3:, :]    # (5, num_coils)
        return J_grad5
    
    def currents_full_jacobian_old(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute the currents -> [field, grad5] Jacobian at a single position.

        Because the model is linear in currents, grad5(pos, currents) = J_grad5_phys(pos) @ currents,
        where J_grad5_phys(pos) is this (5, num_coils) matrix.

        pos:     (3,)              (unnormalized; normalization applied internally)
        returns: (8, num_coils)    in physical units, rows:
                [0] Bx
                [1] By
                [2] Bz
                [3] dBx/dx
                [4] dBx/dy
                [5] dBx/dz
                [6] dBy/dy
                [7] dBy/dz
        """
        self.eval()
        device = next(self.parameters()).device

        # (1, 3) with grad enabled
        pos_t = self._to_tensor(pos, device=device).view(1, -1)
        pos_t.requires_grad_(True)

        # J_phys: (1, 3, num_coils)
        # J_phys[0, 0, j] = Bx contribution of coil j
        # J_phys[0, 1, j] = By contribution of coil j
        # J_phys[0, 2, j] = Bz contribution of coil j
        J_phys = self.forward(pos_t, currents=None)

        num_coils = J_phys.size(-1)
        J_grad5 = torch.empty(5, num_coils, device=device, dtype=J_phys.dtype)

        # We’ll differentiate selected scalar entries of J_phys w.r.t. pos
        for j in range(num_coils):
            # --- Bx_j: J_phys[0, 0, j] ---
            grad_out = torch.zeros_like(J_phys)  # (1, 3, num_coils)
            
            # autograd works with scalars apparently.
            # grad_out acts as a mask or weight tensor in a vector–Jacobian product:
            #   scalar = sum_{c,k} grad_out[0, c, k] * J_phys[0, c, k]
            # Here we put a single 1 at (component=0, coil=j), i.e. Bx_j, and zeros elsewhere:
            #   scalar = 1 * J_phys[0, 0, j] = Bx_j
            # autograd.grad then returns d(Bx_j) / d(pos).
            grad_out[0, 0, j] = 1.0

            (grad_pos,) = torch.autograd.grad(
                outputs=J_phys,
                inputs=pos_t,
                grad_outputs=grad_out,
                retain_graph=True,   # reuse graph for the rest of the loop
                create_graph=False,
            )
            # grad_pos: (1, 3) = [dBx_j/dx, dBx_j/dy, dBx_j/dz]
            J_grad5[0, j] = grad_pos[0, 0]  # dBx/dx
            J_grad5[1, j] = grad_pos[0, 1]  # dBx/dy
            J_grad5[2, j] = grad_pos[0, 2]  # dBx/dz

            # --- By_j: J_phys[0, 1, j] ---
            grad_out.zero_()

            # Now select By_j = J_phys[0, 1, j]
            grad_out[0, 1, j] = 1.0

            (grad_pos2,) = torch.autograd.grad(
                outputs=J_phys,
                inputs=pos_t,
                grad_outputs=grad_out,
                retain_graph=True,
                create_graph=False,
            )
            # grad_pos2: (1, 3) = [dBy_j/dx, dBy_j/dy, dBy_j/dz]
            J_grad5[3, j] = grad_pos2[0, 1]  # dBy/dy
            J_grad5[4, j] = grad_pos2[0, 2]  # dBy/dz

        J_field_np = self._to_numpy(J_phys[0])    # (3, num_coils)
        J_grad5_np = self._to_numpy(J_grad5)    # (5, num_coils)
        J_full = np.vstack([J_field_np, J_grad5_np])  # (8, num_coils)

        return J_full

    def currents_full_jacobian(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute the currents -> [field, grad5] Jacobian at a single position.

        Because the model is linear in currents, grad5(pos, currents) = J_grad5_phys(pos) @ currents,
        where J_grad5_phys(pos) is this (5, num_coils) matrix.

        pos:     (3,)              (unnormalized; normalization applied internally)
        returns: (8, num_coils)    in physical units, rows:
                [0] Bx
                [1] By
                [2] Bz
                [3] dBx/dx
                [4] dBx/dy
                [5] dBx/dz
                [6] dBy/dy
                [7] dBy/dz
        """
        self.eval()
        device = next(self.parameters()).device

        # (1, 3)
        pos_t = self._to_tensor(pos, device=device).view(1, -1)

        # ---- Get J_phys(pos) itself (field Jacobian wrt currents) ----
        # J_phys: (1, 3, num_coils)
        with torch.no_grad():
            J_phys = self.forward(pos_t, currents=None)

        num_coils = J_phys.size(-1)

        # ---- Use jacrev to get d J_phys / d pos ----------------------
        #
        # We want derivatives of J_phys[0, c, j] wrt pos (3D).
        # Define a function of pos only that returns J_phys flattened over (3, num_coils).
        #
        #   pos_vec: (3,) -> J_flat: (3 * num_coils,)
        #
        def J_flat_fn(pos_vec: torch.Tensor) -> torch.Tensor:
            pos_vec = pos_vec.view(1, -1)  # (1, 3)
            J_phys_pos = self.forward(pos_vec, currents=None)  # (1, 3, num_coils)
            return J_phys_pos.view(-1)  # (3 * num_coils,)

        # pos_flat: (3,)
        pos_flat = pos_t.view(-1)

        # jac_flat: (3 * num_coils, 3)
        # row r = d J_flat[r] / d pos[k]
        jac_flat = jacrev(J_flat_fn)(pos_flat)

        # Reshape to (3, num_coils, 3):
        #   first dim: field component (0=Bx, 1=By, 2=Bz)
        #   second dim: coil index
        #   third dim: derivative wrt pos (0=x, 1=y, 2=z)
        jac_J = jac_flat.view(3, num_coils, 3)  # (3, num_coils, 3)

        # ---- Build J_grad5 from selected derivatives -----------------
        # J_grad5: (5, num_coils)
        J_grad5 = torch.empty(5, num_coils, device=device, dtype=J_phys.dtype)

        # For each coil j:
        #  [0] dBx/dx
        #  [1] dBx/dy
        #  [2] dBx/dz
        #  [3] dBy/dy
        #  [4] dBy/dz
        #
        # jac_J[0, j, :] = [dBx_j/dx, dBx_j/dy, dBx_j/dz]
        # jac_J[1, j, :] = [dBy_j/dx, dBy_j/dy, dBy_j/dz]
        J_grad5[0, :] = jac_J[0, :, 0]  # dBx/dx
        J_grad5[1, :] = jac_J[0, :, 1]  # dBx/dy
        J_grad5[2, :] = jac_J[0, :, 2]  # dBx/dz
        J_grad5[3, :] = jac_J[1, :, 1]  # dBy/dy
        J_grad5[4, :] = jac_J[1, :, 2]  # dBy/dz

        # ---- Convert to numpy and stack exactly as before ------------
        J_field_np = self._to_numpy(J_phys[0])   # (3, num_coils)
        J_grad5_np = self._to_numpy(J_grad5)     # (5, num_coils)
        J_full = np.vstack([J_field_np, J_grad5_np])  # (8, num_coils)

        return J_full
    
    def get_currents(self, pos: np.ndarray, target_field: np.ndarray = None, target_grad5: np.ndarray = None) -> np.ndarray:

        # Check if either target_field or target_grad5 is provided
        if target_field is None and target_grad5 is None:
            raise ValueError("At least one of target_field or target_grad5 must be provided.")
        
        # Check validity of input shapes
        if target_field is not None and target_field.shape != (3,):
            raise ValueError(f"target_field must have shape (3,), got {target_field.shape}")
        
        if target_grad5 is not None and target_grad5.shape != (5,):
            raise ValueError(f"target_grad5 must have shape (5,), got {target_grad5.shape}")

        if target_grad5 is None:
            if target_field.shape != (3,):
                raise ValueError(f"target_field must have shape (3,), got {target_field.shape}")

            actuation_mat = self.currents_field_jacobian(pos)  # (3, num_coils)
            pseudo_inv = np.linalg.pinv(actuation_mat)        # (num_coils, 3)
            currents = pseudo_inv @ target_field              # (num_coils,)
            return currents
        
        if target_field is None:
            if target_grad5.shape != (5,):
                raise ValueError(f"target_grad5 must have shape (5,), got {target_grad5.shape}")

            actuation_mat = self.currents_grad5_jacobian(pos)  # (5, num_coils)
            pseudo_inv = np.linalg.pinv(actuation_mat)         # (num_coils, 5)
            currents = pseudo_inv @ target_grad5               # (num_coils,)
            return currents
        
        # Both target_field and target_grad5 are provided
        actuation_mat = self.currents_full_jacobian(pos)  # (8, num_coils)
        pseudo_inv = np.linalg.pinv(actuation_mat)        # (num_coils,
        combined_target = np.concatenate([target_field, target_grad5])  # (8,)
        currents = pseudo_inv @ combined_target           # (num_coils,)
        return currents