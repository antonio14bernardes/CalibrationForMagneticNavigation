import numpy as np
from typing import Tuple
from .calibration import Calibration
from mag_manip.mag_manip import ForwardModelMPEM

class MPEM(Calibration):
    def __init__(self, params_path: str, name: str = "MPEM"):
        
        self.params_path_ = params_path
        self.core_ = ForwardModelMPEM()
        self.core_.setCalibrationFile(params_path)

        num_coils = int(self.core_.getNumCoils())
        super().__init__(name, num_coils=num_coils)

    def get_field(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        return self.core_.computeFieldFromCurrents(pos.astype(np.float64), currents.astype(np.float64)).reshape(-1,3)[0] * 1e3 * self._multiplier  # Default is mT
    
    def get_grad5(self, pos, currents):
        # print(pos.shape, currents.shape)
        return self.core_.computeGradient5FromCurrents(pos.astype(np.float64), currents.astype(np.float64)).reshape(-1,5)[0] * 1e3 * self._multiplier  # Convert from T/m to mT/m
    
    def get_grad9(self, pos: np.ndarray, currents: np.ndarray) -> np.ndarray:
        grad5 = self.get_grad5(pos, currents)
        grad9 = np.zeros((3,3))
        grad9[0,0] = grad5[0]
        grad9[1,1] = grad5[1]
        grad9[2,2] = -(grad5[0] + grad5[1])
        grad9[0,1] = grad5[2]
        grad9[1,0] = grad5[2]
        grad9[0,2] = grad5[3]
        grad9[2,0] = grad5[3]
        grad9[1,2] = grad5[4]
        grad9[2,1] = grad5[4]
        return grad9
    
    
    def get_field_and_grad5(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        field = self.get_field(pos, currents)
        grad5 = self.get_grad5(pos, currents)
        return field, grad5
    
    def get_field_and_grad9(self, pos: np.ndarray, currents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        field = self.get_field(pos, currents)
        grad9 = self.get_grad9(pos, currents)
        return field, grad9
    
    def currents_field_jacobian(self, pos: np.ndarray) -> np.ndarray:
        return self.core_.getFieldActuationMatrix(pos.astype(np.float64)) * 1e3 * self._multiplier  # Convert from T/A to mT/A
    
    def currents_grad5_jacobian(self, pos: np.ndarray) -> np.ndarray:
        full = self.core_.getActuationMatrix(pos.astype(np.float64)) * 1e3 * self._multiplier  # Convert from T/m/A to mT/m/A
        return full[3:, :]  # Extract rows
    
    def currents_full_jacobian(self, pos: np.ndarray) -> np.ndarray:
        return self.core_.getActuationMatrix(pos.astype(np.float64)) * 1e3 * self._multiplier  # Convert from T/A and T/m/A to mT/A and mT/m/A
    
    def currents_field_jacobian_and_bias(self, pos: np.ndarray) -> np.ndarray:
        act_mat = self.core_.getFieldActuationMatrix(pos.astype(np.float64)) * 1e3 * self._multiplier  # Convert from T/A to mT/A
        bias = self.get_field(pos, np.zeros(self.core_.getNumCoils()))  # Get bias field with zero currents
        return act_mat, bias
    
    def currents_grad5_jacobian_and_bias(self, pos: np.ndarray) -> np.ndarray:
        full = self.core_.getActuationMatrix(pos.astype(np.float64)) * 1e3 * self._multiplier  # Convert from T/m/A to mT/m/A
        grad5_mat =  full[3:, :]  # Extract rows
        bias = self.get_grad5(pos, np.zeros(self.core_.getNumCoils()))  # Get bias grad5 with zero currents 
        return grad5_mat, bias
    
    def currents_full_jacobian_and_bias(self, pos: np.ndarray) -> np.ndarray:
        act_mat = self.core_.getActuationMatrix(pos.astype(np.float64)) * 1e3 * self._multiplier  # Convert from T/A and T/m/A to mT/A and mT/m/A
        bias_field = self.get_field(pos, np.zeros(self.core_.getNumCoils()))  # Get bias field with zero currents
        bias_grad5 = self.get_grad5(pos, np.zeros(self.core_.getNumCoils()))  # Get bias grad5 with zero currents
        bias = np.concatenate([bias_field, bias_grad5])  # Combine biases
        return act_mat, bias
    
    def repr(self) -> str:
        return f"MPEM(name={self.name_}, params_path={self.params_path_})"