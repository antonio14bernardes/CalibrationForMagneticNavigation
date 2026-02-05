from . import calibration
from . import nn_calibration

from .calibration import Calibration
from .nn_calibration import NNCalibration

from .linear_net import LinearNet
from .actuation_net import ActuationNet
from .direct_net import DirectNet
from .potential_net import PotentialNet
from .constants import OCTOMAG_EMS, NAVION_EMS

from .gbt_callibration import GBTCalibration
from .direct_gbt import DirectGBT

# ---- MPEM ----
MPEM_AVAILABLE = False
MPEM = None

try:
    from .mpem import MPEM as _MPEM
    MPEM = _MPEM
    MPEM_AVAILABLE = True
except ImportError:
    MPEM = None
    MPEM_AVAILABLE = False