import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class Calibration(ABC):
    def __init__(self, name: str, num_coils: int = 8):
        self._name = name
        self._multiplier = 1.0
        self._num_coils = num_coils

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def num_coils(self) -> int:
        return int(self._num_coils)
    
    def mT_to_T(self):
        self._multiplier = 1e-3

    @abstractmethod
    def get_field(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement get_field method.")
    
    @abstractmethod
    def get_grad5(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement get_grad5 method.")

    @abstractmethod
    def get_field_and_grad9(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclasses must implement get_field_and_grad9 method.")
    
    @abstractmethod
    def get_field_and_grad5(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclasses must implement get_field_and_grad5 method.")
    
    @abstractmethod
    def get_grad9(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement get_grad9 method.")
    
    @abstractmethod
    def currents_field_jacobian(self, pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement currents_field_jacobian method.")
    
    @abstractmethod
    def currents_grad5_jacobian(self, pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement currents_grad5_jacobian method.")
    
    @abstractmethod
    def currents_full_jacobian(self, pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement currents_full_jacobian method.")
    

    def _zero_currents(self) -> np.ndarray:
        return np.zeros((self.num_coils,), dtype=float)

    def currents_field_jacobian_and_bias(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Default:
          J = currents_field_jacobian(pos)
          bias = get_field(pos, zeros)
        """
        J = self.currents_field_jacobian(pos)
        bias = self.get_field(pos, self._zero_currents())
        bias = np.asarray(bias, dtype=float).reshape(-1)
        return J, bias

    def currents_grad5_jacobian_and_bias(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Default:
          J = currents_grad5_jacobian(pos)
          bias = get_grad5(pos, zeros)
        """
        J = self.currents_grad5_jacobian(pos)
        bias = self.get_grad5(pos, self._zero_currents())
        bias = np.asarray(bias, dtype=float).reshape(-1)
        return J, bias

    def currents_full_jacobian_and_bias(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Default:
          J = currents_full_jacobian(pos)
          bias = [get_field(pos, zeros); get_grad5(pos, zeros)]
        """
        J = self.currents_full_jacobian(pos)
        z = self._zero_currents()
        field0 = np.asarray(self.get_field(pos, z), dtype=float).reshape(-1)
        grad50 = np.asarray(self.get_grad5(pos, z), dtype=float).reshape(-1)
        full_bias = np.concatenate([field0, grad50], axis=0)
        return J, full_bias
    
    def get_currents(self, pos, target_field=None, target_grad5=None):
        """
        Solve for currents using pseudoinverse, assuming the model is affine in currents:
            [field; grad5](pos, I) = J_full(pos) @ I + bias_full(pos)

        Subclasses that do not satisfy this assumption MUST override and raise NotImplementedError.
        """
        if target_field is None and target_grad5 is None:
            raise ValueError("Provide target_field and/or target_grad5")

        if target_field is not None:
            target_field = np.asarray(target_field, dtype=float).reshape(-1)
            if target_field.shape != (3,):
                raise ValueError(f"target_field must have shape (3,), got {target_field.shape}")

        if target_grad5 is not None:
            target_grad5 = np.asarray(target_grad5, dtype=float).reshape(-1)
            if target_grad5.shape != (5,):
                raise ValueError(f"target_grad5 must have shape (5,), got {target_grad5.shape}")

        if target_grad5 is None:
            J, b = self.currents_field_jacobian_and_bias(pos)
            return np.linalg.pinv(J) @ (target_field - b)

        if target_field is None:
            J_full, b_full = self.currents_full_jacobian_and_bias(pos)
            Jg = J_full[3:, :]
            bg = b_full[3:]
            return np.linalg.pinv(Jg) @ (target_grad5 - bg)

        J_full, b_full = self.currents_full_jacobian_and_bias(pos)
        y = np.concatenate([target_field, target_grad5])
        return np.linalg.pinv(J_full) @ (y - b_full)
    

    
    def repr(self) -> str:
        return f"Calibration(name={self.name})"