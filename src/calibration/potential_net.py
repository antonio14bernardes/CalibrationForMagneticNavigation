import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, vmap
from typing import Optional, Tuple
import warnings

from .nn_calibration import NNCalibration


class PotentialNet(NNCalibration):
    """
    Curl-free-by-construction calibration:
      - predicts scalar potentials phi_k(pos) for each coil + bias potential phi0(pos)
      - field is gradient: B = ∇phi0 + Σ I_k ∇phi_k
    """

    def __init__(
        self,
        hidden_dims=(150, 100, 50),
        num_coils: int = 8,
        name: str = "PotentialNet",
        position_normalization_dict: Optional[dict] = None,
        field_normalization_dict: Optional[dict] = None,
    ):
        
        # We need to ensure that the std is the same on all field components for curl-free property to hold after denormalization
        if field_normalization_dict is not None:
            stds = field_normalization_dict["std"]
            if not (np.isclose(stds[0], stds[1]) and np.isclose(stds[0], stds[2])):
                new_std = float(np.mean(stds))
                field_normalization_dict["std"] = [new_std, new_std, new_std]
                warnings.warn("For PotentialNet, under the current structure, field normalization std must be isotropic over components to preserve curl-free property.\n"
                              f"We will override the std to be the average value of the three components.\nNew normalization dictionary: {field_normalization_dict}\n"
                              "Not the most elegant solution, but is reasonable given our datasets.")
        super().__init__(
            name=name,
            num_coils=num_coils,
            position_normalization_dict=position_normalization_dict,
            field_normalization_dict=field_normalization_dict,
        )
        self.hidden_dims = tuple(hidden_dims)

        input_dim = 3
        output_dim = self.num_coils + 1  # phi_0..phi_{Nc-1} plus bias phi_bias

        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def get_config(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "num_coils": self.num_coils,
            "name": self.name,
            "position_normalization_dict": self.position_normalization_dict,
            "field_normalization_dict": self.field_normalization_dict,
        }

    # -----------------------------
    # core: potentials + derivatives
    # -----------------------------

    def _phi_single(self, pos_phys_3: torch.Tensor) -> torch.Tensor:
        # pos_phys_3: (3,)
        pos = pos_phys_3.view(1, 3)
        pos_norm = self._normalize_input(pos)
        return self.mlp(pos_norm).view(-1)  # (Nc+1,)

    def _Jb_from_pos(self, pos_phys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns full (B,3,Nc) and (B,3)
        if pos_phys.dim() == 1:
            pos_phys = pos_phys.view(1, 3)

        grads = vmap(jacrev(self._phi_single))(pos_phys)  # (B, Nc+1, 3)

        J_field = -grads[:, :self.num_coils, :].transpose(1, 2).contiguous()  # (B,3,Nc)
        b_field = -grads[:, self.num_coils, :]                                # (B,3)
        return J_field, b_field

    def _field_from_pos_currents(self, pos_phys, currents) -> torch.Tensor:
        single = (pos_phys.dim() == 1)
        if single:
            pos_phys = pos_phys.view(1, 3)

        if currents.dim() == 1:
            currents = currents.view(1, -1)
        if currents.size(0) == 1 and pos_phys.size(0) > 1:
            currents = currents.expand(pos_phys.size(0), -1)

        pos_req = pos_phys.detach().requires_grad_(True)

        pos_norm = self._normalize_input(pos_req)
        phi = self.mlp(pos_norm)

        phi_coils = phi[:, :self.num_coils]
        phi_bias  = phi[:, self.num_coils]
        phi_tot = (phi_coils * currents).sum(dim=1) + phi_bias

        grad_phi = torch.autograd.grad(
            outputs=phi_tot,
            inputs=pos_req,
            grad_outputs=torch.ones_like(phi_tot),
            create_graph=True,
            retain_graph=True,
        )[0]

        field = -grad_phi
        return field[0] if single else field

    # forward

    def forward(self, pos: torch.Tensor, currents: Optional[torch.Tensor] = None, return_Jb: bool = True):
        """
        If currents is None:
            returns (J_phys, b_phys)
        else:
            returns (field_phys, J_phys, b_phys) if return_Jb else field_phys
        """
        with torch.enable_grad():
            if currents is None:
                # (optional) implement J,b later; for now just compute them via autograd loop if you need
                J_norm, b_norm = self._Jb_from_pos(pos)  # (B,3,Nc), (B,3)

                if self.field_normalization_dict is None:
                    J_phys, b_phys = J_norm, b_norm
                else:
                    mean = torch.as_tensor(self.field_normalization_dict["mean"], device=pos.device, dtype=pos.dtype)
                    std  = torch.as_tensor(self.field_normalization_dict["std"],  device=pos.device, dtype=pos.dtype)
                    J_phys = J_norm * std.view(1, 3, 1)
                    b_phys = b_norm * std + mean.view(1, 3)

                return J_phys * self._multiplier, b_phys * self._multiplier

            # currents provided -> field
            field_norm = self._field_from_pos_currents(pos, currents)

            if self.field_normalization_dict is None:
                field_phys = field_norm
            else:
                mean = torch.as_tensor(self.field_normalization_dict["mean"], device=pos.device, dtype=pos.dtype)
                std  = torch.as_tensor(self.field_normalization_dict["std"],  device=pos.device, dtype=pos.dtype)
                field_phys = field_norm * std + mean

            field_phys = field_phys * self._multiplier

            if not return_Jb:
                return field_phys

            # If you want J,b for logging, compute them
            J_norm, b_norm = self._Jb_from_pos(pos)
            if self.field_normalization_dict is None:
                J_phys, b_phys = J_norm, b_norm
            else:
                mean = torch.as_tensor(self.field_normalization_dict["mean"], device=pos.device, dtype=pos.dtype)
                std  = torch.as_tensor(self.field_normalization_dict["std"],  device=pos.device, dtype=pos.dtype)
                J_phys = J_norm * std.view(1, 3, 1)
                b_phys = b_norm * std + mean.view(1, 3)

            return field_phys, J_phys * self._multiplier, b_phys * self._multiplier

    # -----------------------------
    # Calibration (numpy) API
    # -----------------------------

    def get_field(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device

        pos_t = self._to_tensor(pos, device=device)
        cur_t = self._to_tensor(currents, device=device)

        field, _, _ = self.forward(pos_t, cur_t)
        out = self._to_numpy(field)
        return out[0] if out.shape[0] == 1 else out

    def get_field_and_grad9(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-sample:
        returns field (3,), grad9 (3,3) where grad9[i,j] = dB_i / dpos_j
        """
        self.eval()
        device = next(self.parameters()).device

        # shapes: pos (3,), currents (Nc,)
        pos_t = self._to_tensor(pos, device=device).view(-1)          # (3,)
        cur_t = self._to_tensor(currents, device=device).view(-1)     # (Nc,)

        Nc = self.num_coils

        # g: (Nc+1, 3)  where g[k] = ∇phi_k, and g[Nc] = ∇phi_bias
        g = jacrev(self._phi_single)(pos_t)

        # H: (Nc+1, 3, 3) where H[k] = Hessian(phi_k), H[Nc] = Hessian(phi_bias)
        H = jacrev(jacrev(self._phi_single))(pos_t)

        # field in "normalized field units" (same units as your current forward before denorm)
        field_norm = -(g[:Nc].T @ cur_t + g[Nc])  # (3,)

        # grad9 in normalized units: sum_k I_k * Hess(phi_k) + Hess(phi_bias)
        grad9_norm = -(H[:Nc] * cur_t.view(Nc, 1, 1)).sum(dim=0) - H[Nc]  # (3,3)

        # denormalize field + scale grad9 correctly
        if self.field_normalization_dict is not None:
            mean = torch.as_tensor(self.field_normalization_dict["mean"], device=device, dtype=pos_t.dtype)  # (3,)
            std  = torch.as_tensor(self.field_normalization_dict["std"],  device=device, dtype=pos_t.dtype)  # (3,)

            field_phys = field_norm * std + mean

            # If B_phys[i] = std[i] * B_norm[i] + mean[i],
            # then dB_phys[i]/dx = std[i] * dB_norm[i]/dx  (mean vanishes)
            grad9_phys = grad9_norm * std.view(3, 1)
        else:
            field_phys = field_norm
            grad9_phys = grad9_norm

        # multiplier applies to field and its derivatives the same way
        field_phys = field_phys * self._multiplier
        grad9_phys = grad9_phys * self._multiplier

        field_np = self._to_numpy(field_phys.view(1, 3))[0]
        grad9_np = self._to_numpy(grad9_phys)

        return field_np, grad9_np

    def get_grad9(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        return self.get_field_and_grad9(pos, currents)[1]

    def get_field_and_grad5(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        field, grad9 = self.get_field_and_grad9(pos, currents)
        dbx_dxyz = grad9[0, :]
        dby_dyz  = grad9[1, 1:]
        grad5 = np.concatenate([dbx_dxyz, dby_dyz])
        return field, grad5

    def get_grad5(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        return self.get_field_and_grad5(pos, currents)[1]

    def currents_field_jacobian_and_bias(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (J_field, b_field) in physical units:
          J_field: (3,Nc)
          b_field: (3,)
        """
        self.eval()
        device = next(self.parameters()).device
        pos_t = self._to_tensor(pos, device=device).view(1, 3)

        J, b = self.forward(pos_t, currents=None)  # (1,3,Nc), (1,3)
        return self._to_numpy(J)[0], self._to_numpy(b)[0]

    def currents_field_jacobian(self, pos: np.ndarray) -> np.ndarray:
        return self.currents_field_jacobian_and_bias(pos)[0]

    # -----------------------------
    # Fast closed-form construction of [field; grad5] wrt currents from Hessians
    # -----------------------------

    def currents_full_jacobian_and_bias(self, pos: np.ndarray):
        """
        Returns:
        J_full:    (8, Nc)
        full_bias: (8,)
        in physical units.
        """
        import torch
        from torch.func import jacrev

        self.eval()
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype

        pos_t = self._to_tensor(pos, device=device, dtype=dtype).view(-1)  # (3,)
        Nc = self.num_coils

        # derivatives (these require grad internally; that's fine)
        g = jacrev(self._phi_single)(pos_t)                 # (Nc+1, 3)
        H = jacrev(jacrev(self._phi_single))(pos_t)         # (Nc+1, 3, 3)

        g_coils = g[:Nc, :]          # (Nc,3)
        g_bias  = g[Nc, :]           # (3,)
        H_coils = H[:Nc, :, :]       # (Nc,3,3)
        H_bias  = H[Nc, :, :]        # (3,3)

        # Field wrt currents and bias (match your PotentialNet convention: B = -∇phi_total)
        J_field = -g_coils.T         # (3,Nc)
        b_field = -g_bias            # (3,)

        # grad5 rows from Hessians:
        # [dBx/dx, dBx/dy, dBx/dz, dBy/dy, dBy/dz]
        J_grad5 = -torch.stack(
            [
                H_coils[:, 0, 0],
                H_coils[:, 0, 1],
                H_coils[:, 0, 2],
                H_coils[:, 1, 1],
                H_coils[:, 1, 2],
            ],
            dim=0,
        )  # (5, Nc)

        bias_grad5 = -torch.stack(
            [
                H_bias[0, 0],
                H_bias[0, 1],
                H_bias[0, 2],
                H_bias[1, 1],
                H_bias[1, 2],
            ],
            dim=0,
        )  # (5,)

        # normalization (do it in torch)
        if self.field_normalization_dict is not None:
            mean = torch.as_tensor(self.field_normalization_dict["mean"], device=device, dtype=dtype)  # (3,)
            std  = torch.as_tensor(self.field_normalization_dict["std"],  device=device, dtype=dtype)  # (3,)

            # field scaling: B_phys = std * B_norm + mean  => J scales by std, bias scales by std and +mean
            J_field = J_field * std.view(3, 1)
            b_field = b_field * std + mean

            # grad rows scale by the std of the component they belong to (Bx rows use std[0], By rows use std[1])
            row_scale = torch.stack([std[0], std[0], std[0], std[1], std[1]])  # (5,)
            J_grad5 = J_grad5 * row_scale.view(5, 1)
            bias_grad5 = bias_grad5 * row_scale

        # multiplier applies to everything
        mult = torch.as_tensor(self._multiplier, device=device, dtype=dtype)
        J_field     = J_field * mult
        b_field     = b_field * mult
        J_grad5     = J_grad5 * mult
        bias_grad5  = bias_grad5 * mult

        J_full    = torch.cat([J_field, J_grad5], dim=0)          # (8, Nc)
        full_bias = torch.cat([b_field, bias_grad5], dim=0)        # (8,)

        # convert once at the end
        return (
            J_full.detach().cpu().numpy(),
            full_bias.detach().cpu().numpy(),
        )
    
    def currents_grad5_jacobian(self, pos: np.ndarray) -> np.ndarray:
        """
        Returns J_grad5 (5, Nc) wrt currents at a single position.
        """
        J_full, _ = self.currents_full_jacobian_and_bias(pos)
        return J_full[3:, :]   # (5, Nc)

    def currents_full_jacobian(self, pos: np.ndarray) -> np.ndarray:
        """
        Returns J_full (8, Nc) wrt currents at a single position.
        """
        J_full, _ = self.currents_full_jacobian_and_bias(pos)
        return J_full