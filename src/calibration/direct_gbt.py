import numpy as np
import pandas as pd
from typing import Tuple

from . import GBTCalibration
from .calibration import Calibration
from .constants import OCTOMAG_EMS

class DirectGBT(GBTCalibration):
    def __init__(self, name: str = "GBT_CALIBRATION", current_names: list = OCTOMAG_EMS):
        target_names = ["Bx", "By", "Bz"]
        position_names = ["x", "y", "z"]

        super().__init__(name, target_names, position_names, current_names)
    
    def get_field(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        return self.predict_targets(pos, currents)
    
    def get_grad9(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        return self.compute_grads_wrt_position(pos, currents)

    def get_grad5(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        J = self.get_grad9(pos, currents)

        if J.ndim == 2:
            return np.array([J[0,0], J[1,0], J[2,0], J[1,1], J[1,2]], dtype=float)

        return np.stack([J[:,0,0], J[:,1,0], J[:,2,0], J[:,1,1], J[:,1,2]], axis=1)

    def get_field_and_grad9(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        field = self.get_field(pos, currents)
        grad9 = self.get_grad9(pos, currents)
        return field, grad9
    
    
    def get_field_and_grad5(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        field = self.get_field(pos, currents)
        grad5 = self.get_grad5(pos, currents)
        return field, grad5
    
    def get_currents(self, pos: np.ndarray, target_field: np.ndarray = None, target_grad5: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement get_currents method.")
    
    def currents_field_jacobian(self, pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement currents_field_jacobian method.")
    
    def currents_grad5_jacobian(self, pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement currents_grad5_jacobian method.")
    
    def currents_full_jacobian(self, pos: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement currents_full_jacobian method.")
    
    def repr(self) -> str:
        return f"DirectGBT(name={self.name})"