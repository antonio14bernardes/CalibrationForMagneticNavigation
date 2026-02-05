import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev
from typing import Optional, Tuple


from .nn_calibration import NNCalibration


class ActuationNet(NNCalibration):
    def __init__(
        self,
        hidden_dims=(150, 100, 50),
        num_coils: int = 8,
        name: str = "ActuationNet",
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
        output_dim = 3 * self.num_coils + 3  # 24 + 3

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

    # ----- torch forward, everything in physical units -----

    def forward(self, pos: torch.Tensor, currents: Optional[torch.Tensor] = None):
        """
        pos:      (batch, 3)   (unnormalized; will be normalized internally)
        currents: (batch, 8) or (8,) or None

        If currents is None:
            returns (J_phys, b_phys)

        else:
            returns (field_phys, J_phys, b_phys)
        """
        # normalize positions if dict is set
        pos = self._normalize_input(pos)

        flat = self.mlp(pos)                        # (batch, 27)

        J_flat = flat[..., : 3 * self.num_coils]      # (batch, 24)
        b_norm = flat[..., 3 * self.num_coils :]      # (batch, 3)
        J_norm = J_flat.view(-1, 3, self.num_coils)   # (batch, 3, 8)

        # --- convert J, b to physical units if field normalization exists ---
        if self.field_normalization_dict is None:
            J_phys = J_norm
            b_phys = b_norm
        else:
            mean = torch.as_tensor(
                self.field_normalization_dict["mean"],
                device=J_norm.device,
                dtype=J_norm.dtype,
            )  # (3,)
            std = torch.as_tensor(
                self.field_normalization_dict["std"],
                device=J_norm.device,
                dtype=J_norm.dtype,
            )  # (3,)

            # J_phys: each output dimension scaled by std
            J_phys = J_norm * std.view(1, 3, 1)     # (batch, 3, 8)

            # b_phys: denormalize like the field would be
            b_phys = b_norm * std + mean            # (batch, 3)

        J_phys *= self._multiplier
        b_phys *= self._multiplier

        if currents is None:
            return J_phys, b_phys

        # ensure batch shape on currents
        if currents.dim() == 1:
            currents = currents.unsqueeze(0).expand(J_phys.size(0), -1)  # (batch, 8)

        # field in physical units
        field_phys = torch.einsum("bij,bj->bi", J_phys, currents) + b_phys  # (batch, 3)

        return field_phys, J_phys, b_phys

    # ----- Calibration (numpy) API -----
    def get_field(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        """
        pos:      (N, 3)        (unnormalized)
        currents: (8,) or (N, 8)
        returns:  (N, 3)        (physical units)
        """
        self.eval()
        device = next(self.parameters()).device

        pos_t = self._to_tensor(pos, device=device)
        cur_t = self._to_tensor(currents, device=device)

        with torch.no_grad():
            field, _, _ = self.forward(pos_t, cur_t)

        field_np = self._to_numpy(field)

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
    
    def get_field_and_grad9(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        pos:      (3,)          (unnormalized)
        currents: (num_coils,)  == (8,)

        returns:
            - field: (3,)    in physical units
            - grad9: (3, 3) Jacobian d(field)/d(pos),
                    grad9[i, j] = d field[i] / d pos[j]
        """
        self.eval()
        device = next(self.parameters()).device

        # to torch
        pos_t = self._to_tensor(pos, device=device)
        cur_t = self._to_tensor(currents, device=device)

        # add batch dimension
        pos_t = pos_t.view(1, -1)
        cur_t = cur_t.view(1, -1)

        # ----- define function field(pos_vec) with currents fixed -----
        def field_wrt_pos(pos_vec: torch.Tensor) -> torch.Tensor:
            """
            pos_vec: (3,)
            returns: (3,) field at that position, with fixed currents
            """
            pos_vec = pos_vec.view(1, -1)
            field, _, _ = self.forward(pos_vec, cur_t)
            return field.view(-1)

        pos_flat = pos_t.view(-1)

        # ---- Jacobian d(field)/d(pos) ----
        # jacobian: (3, 3) where jacobian[i, j] = d field[i] / d pos[j]
        jacobian = jacrev(field_wrt_pos)(pos_flat)

        # ---- field value itself (no grad needed) ----
        with torch.no_grad():
            field, _, _ = self.forward(pos_t, cur_t)

        # --- convert to numpy ---
        field_np = self._to_numpy(field)[0]
        grad9_np = self._to_numpy(jacobian)

        return field_np, grad9_np

    def currents_field_jacobian(self, pos: np.ndarray) -> np.ndarray:
            """
            Jacobian wrt currents in physical units.

            pos:     (3,)       (unnormalized)
            returns: (3, 8)
            """
            J, _ = self.currents_field_jacobian_and_bias(pos)  # to ensure eval mode
            return J
    
    def currents_field_jacobian_and_bias(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns both the Jacobian wrt currents and the bias term, in physical units.

        pos:     (3,)
        returns: tuple of
                - J: (3, 8)
                - b: (3,)
        """
        self.eval()
        device = next(self.parameters()).device

        pos_t = self._to_tensor(pos, device=device)
        pos_t = pos_t.view(1, -1)

        with torch.no_grad():
            J_phys, b_phys = self.forward(pos_t, currents=None)  # (1, 3, 8), (1, 3)

        J_np = self._to_numpy(J_phys)[0]
        b_np = self._to_numpy(b_phys)[0]

        return J_np, b_np

    def currents_grad5_jacobian(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian of grad5 w.r.t. currents at a single position.

        Model:
            G(x, I) = J_grad5(x) @ I + G0(x)

        This method returns J_grad5(x), i.e. the linear map from currents to grad5.

        Args:
            pos: (3,) position in physical units (unnormalized; normalization applied internally)

        Returns:
            J_grad5: (5, num_coils) array
                Row ordering matches grad5 definition:
                    [0] dBx/dx
                    [1] dBx/dy
                    [2] dBx/dz
                    [3] dBy/dy
                    [4] dBy/dz
        """
        J_grad5, _ = self.currents_grad5_jacobian_and_bias(pos)
        return J_grad5


    def currents_grad5_jacobian_and_bias(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both the Jacobian of grad5 w.r.t. currents and the grad5 bias G0 at a position.

        Gradient model at position x:
            G(x, I) = J_grad5(x) @ I + G0(x)

        This method extracts J_grad5(x) and G0(x) from the full
        [field; grad5] Jacobian and bias:

            [field; grad5](x, I) = full_bias(x) + J_full(x) @ I

        Args:
            pos: (3,) position in physical units (unnormalized; normalization applied internally)

        Returns:
            J_grad5: (5, num_coils) array
                Jacobian of grad5 w.r.t. currents.
            bias_grad5: (5,) array
                G0(x), the grad5 offset at zero currents, with components:
                    [0] dB0x/dx
                    [1] dB0x/dy
                    [2] dB0x/dz
                    [3] dB0y/dy
                    [4] dB0y/dz
        """
        J_full, full_bias = self.currents_full_jacobian_and_bias(pos)
        J_grad5 = J_full[3:, :]
        bias_grad5 = full_bias[3:]
        return J_grad5, bias_grad5


    def currents_full_jacobian(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute the full Jacobian of [field; grad5] w.r.t. currents at a position.

        Combined model:
            [field; grad5](x, I) = full_bias(x) + J_full(x) @ I

        where the stacked output is:
            [field; grad5] = [Bx, By, Bz, dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz]^T

        This method returns only J_full(x), ignoring the bias term.

        Args:
            pos: (3,) position in physical units (unnormalized; normalization applied internally)

        Returns:
            J_full: (8, num_coils) array
                Rows correspond to:
                    [0] Bx sensitivity wrt currents
                    [1] By
                    [2] Bz
                    [3] dBx/dx
                    [4] dBx/dy
                    [5] dBx/dz
                    [6] dBy/dy
                    [7] dBy/dz
        """
        J, _ = self.currents_full_jacobian_and_bias(pos)
        return J

    def currents_full_jacobian_and_bias(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast version using torch.func (jacrev), avoiding per-coil autograd.grad loops.

        Returns:
            J_full:    (8, Nc)   rows = [Bx,By,Bz,dBx/dx,dBx/dy,dBx/dz,dBy/dy,dBy/dz] wrt currents
            full_bias: (8,)      same stacked outputs at zero currents
        """
        import torch
        from torch.func import jacrev

        self.eval()
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        Nc = self.num_coils

        # --- pos as tensor (3,)
        pos_t = self._to_tensor(pos, device=device, dtype=dtype).view(-1)

        # --- helper: return J_field(pos) and b_field(pos) in TORCH, physical units ---
        # forward(pos, currents=None) already returns physical units and includes multiplier.
        def Jb_of_pos(p: torch.Tensor):
            # p: (3,)
            J, b = self.forward(p.view(1, 3), currents=None)   # J: (1,3,Nc), b: (1,3)
            return J[0], b[0]                                  # (3,Nc), (3,)

        # Compute values
        J_field, b_field = Jb_of_pos(pos_t)                    # (3,Nc), (3,)

        # Compute derivatives wrt position in ONE jacrev call:
        # dJ_field/dpos has shape (3,Nc,3) because output (3,Nc), input (3,)
        dJ_dpos = jacrev(lambda p: Jb_of_pos(p)[0])(pos_t)     # (3, Nc, 3)

        # Bias field Jacobian wrt pos: (3,3)
        db_dpos = jacrev(lambda p: Jb_of_pos(p)[1])(pos_t)     # (3, 3)

        # Build grad5 wrt currents (5,Nc) from dJ/dpos:
        # Bx sensitivity is row 0, By sensitivity is row 1
        # dBx/dx = dJ[0,:,0], dBx/dy = dJ[0,:,1], dBx/dz = dJ[0,:,2]
        # dBy/dy = dJ[1,:,1], dBy/dz = dJ[1,:,2]
        J_grad5 = torch.stack(
            [
                dJ_dpos[0, :, 0],
                dJ_dpos[0, :, 1],
                dJ_dpos[0, :, 2],
                dJ_dpos[1, :, 1],
                dJ_dpos[1, :, 2],
            ],
            dim=0,
        )  # (5, Nc)

        # Build bias grad5 (5,) from db/dpos:
        bias_grad5 = torch.stack(
            [
                db_dpos[0, 0],
                db_dpos[0, 1],
                db_dpos[0, 2],
                db_dpos[1, 1],
                db_dpos[1, 2],
            ],
            dim=0,
        )  # (5,)

        # Stack full outputs
        J_full = torch.cat([J_field, J_grad5], dim=0)          # (8, Nc)
        full_bias = torch.cat([b_field, bias_grad5], dim=0)    # (8,)

        return (
            J_full.detach().cpu().numpy(),
            full_bias.detach().cpu().numpy(),
        )