"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Strategy(ABC):
    """
    Every strategy must implement generate_weights().

    The engine calls it once per trading day, passing the full price history
    up to and including that day.  The returned weight array must have the
    same length as prices.columns (one entry per ticker).
    """

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> np.ndarray:
        """
        Parameters
        ----------
        prices : pd.DataFrame
            Shape (T × N): T trading days of history (today is the last row),
            N tickers aligned to the engine's ticker list.

        Returns
        -------
        np.ndarray, shape (N,)
            Target allocation fractions. Must be non-negative; the engine
            will normalise if the sum exceeds 1.
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__
