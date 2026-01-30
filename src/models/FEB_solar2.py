# plants/solar.py
from __future__ import annotations
from typing import Union, Iterable, List, Callable
from datetime import datetime
from base import PowerPlant, PlantType

class SolarPlant(PowerPlant):
    """
    Solar outputs according to a 1-second profile (MW).
    profile_source can be:
      - iterable of floats
      - function f(step_index:int)->float
    After profile ends, power = 0.
    """
    def __init__(self, name: str, profile_source: Union[Iterable[float], List[float], Callable[[int], float]], api=None):
        super().__init__(name, PlantType.SOLAR, api)
        self._source = profile_source
        self._it = None
        self._index = 0
        if callable(profile_source):
            self._mode = "callable"
        elif hasattr(profile_source, "__iter__"):
            self._mode = "iter"
            self._it = iter(profile_source)
        else:
            raise ValueError("profile_source must be iterable or callable")

    def step(self, now: datetime, dt_s: float) -> float:
        value = 0.0
        if self._mode == "callable":
            value = float(self._source(self._index))
        else:
            try:
                value = float(next(self._it))
            except StopIteration:
                value = 0.0
        self.current_power_mw = max(0.0, value)
        self._index += 1
        return self.current_power_mw