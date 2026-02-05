import re
from calibration import Calibration, NNCalibration, GBTCalibration, MPEM
import numpy as np

class ModelPhantom:
    def __init__(self, name, dataset_percentage=None, structure=None):
        self._name = name

        # Dataset percentage must be either None or between 0 and 100
        if dataset_percentage is not None and (dataset_percentage < 0 or dataset_percentage > 100):
            raise ValueError("dataset_percentage must be between 0 and 100. None indicates 100%.")
        self._dataset_percentage = np.clip(int(dataset_percentage), 0, 100) if dataset_percentage is not None else 100

        # Structure is either None or a tuple of integers
        if structure is not None:
            if not isinstance(structure, tuple) or not all(isinstance(s, int) for s in structure):
                raise ValueError("structure must be a tuple of integers or None.")
        self._structure = structure

    @property
    def reduced(self):
        return self._dataset_percentage < 100
    @property
    def dataset_percentage(self):
        return self._dataset_percentage
    @property
    def name(self):
        return self._name
    @property
    def structure(self):
        return self._structure

    @classmethod
    def from_calibration(cls, model):
        # Check if model inherits from Calibration
        if not isinstance(model, Calibration):
            raise ValueError("'from_calibration' method can only load objects that inherit from Calibration.")
        
        if isinstance(model, NNCalibration):
            return cls._from_net(model)
        elif isinstance(model, GBTCalibration):
            return cls._from_gbt(model)
        elif isinstance(model, MPEM):
            return cls._from_mpem(model)
        else:
            raise RuntimeError("Could not find a matching model type.")


    @staticmethod
    def _split_base_and_percentage(prefix: str):
        """
        If prefix ends with _<digits>, treat that as dataset_percentage.
        Otherwise dataset_percentage = 100.
        """
        m = re.match(r"^(?P<base>.+)_(?P<pct>\d{1,3})$", prefix)
        if m:
            base = m.group("base")
            pct = int(m.group("pct"))
            if not (0 <= pct <= 100):
                raise ValueError(f"Parsed dataset percentage {pct} out of range (0..100) from '{prefix}'.")
            return base, pct
        return prefix, 100

    @classmethod
    def _from_net(cls, nn):
        """
        Expected nn.name formats:
          - NameBase_<structure>
          - NameBase_<percentageint>_<structure>
        structure like: 512x512, 512x512x512, 256
        """
        name = str(nn.name)

        # parse trailing _<structure>
        m = re.search(r"_(\d+(?:x\d+)*)$", name)
        if not m:
            raise ValueError(
                f"Could not parse structure from model.name='{name}'. "
                "Expected suffix like '_512x512' or '_512x512x512'."
            )

        structure_str = m.group(1)
        structure = tuple(int(s) for s in structure_str.split("x"))

        # everything before _<structure>
        prefix = name[: m.start()]

        # parse optional dataset percentage from the remaining prefix
        base_name, pct = cls._split_base_and_percentage(prefix)

        return cls(name=base_name, dataset_percentage=pct, structure=structure)

    @classmethod
    def _from_gbt(cls, gbt):
        return cls._from_net(gbt)

    @classmethod
    def _from_mpem(cls, mpem):
        """
          - NameBase                          -> pct=100, structure=(1,)
          - NameBase_<pct>                    -> pct=<pct>, structure=(1,)
          - NameBase_<structure>              -> pct=100, structure=<structure>
          - NameBase_<pct>_<structure>        -> pct=<pct>, structure=<structure>

        Ambiguity rule for a single trailing "_<digits>":
          - if digits <= 100  => treat as pct, structure=(1,)
          - if digits >  100  => treat as structure, pct=100

        structure examples: 1, 2, 3 as in first, second, and third order
        """
        name = str(mpem.name)

        # Try: NameBase_<pct>_<structure>
        m = re.match(r"^(?P<base>.+)_(?P<pct>\d{1,3})_(?P<struct>\d+(?:x\d+)*)$", name)
        if m:
            base = m.group("base")
            pct = int(m.group("pct"))
            if not (0 <= pct <= 100):
                raise ValueError(f"Parsed dataset percentage {pct} out of range (0..100) from '{name}'.")
            struct = tuple(int(s) for s in m.group("struct").split("x"))
            return cls(name=base, dataset_percentage=pct, structure=struct)

        # Try: NameBase_<something> (could be pct OR structure)
        m = re.match(r"^(?P<base>.+)_(?P<tail>\d+(?:x\d+)*)$", name)
        if m:
            base = m.group("base")
            tail = m.group("tail")

            # If it has an 'x' it is definitely a structure
            if "x" in tail:
                struct = tuple(int(s) for s in tail.split("x"))
                return cls(name=base, dataset_percentage=100, structure=struct)

            # Otherwise it's digits: decide pct vs structure by threshold
            val = int(tail)
            if 0 <= val <= 100:
                # treat as pct, and FALL BACK to structure=(1,)
                return cls(name=base, dataset_percentage=val, structure=(1,))
            else:
                # treat as structure
                return cls(name=base, dataset_percentage=100, structure=(val,))

        # No suffix at all: FALL BACK to structure=(1,)
        return cls(name=name, dataset_percentage=100, structure=(1,))

    def string(self, verbose=True):
        if not verbose:
            return self.name


        if self.structure is None:
            structure_str = ""
        else:
            structure_str = "x".join(str(x) for x in self.structure)

        return self.name + "_" + str(self.dataset_percentage) + ("_" if structure_str else "") + structure_str

    def keys(self):
        return (self.name, self.dataset_percentage, self.structure)
    
    def repr(self):
        return f"ModelPhantom(name='{self.name}', dataset_percentage={self.dataset_percentage}, structure={self.structure})"