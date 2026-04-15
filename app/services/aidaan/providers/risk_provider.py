"""
Institutional Risk Data Providers - AIDAAN RAG Interface
========================================================
This module defines the architectural interface for risk data retrieval 
and provides a standard JSON-based implementation for local showcases.

The Provider Pattern used here ensures that shifting from a local JSON 
dictionary to a production Google BigQuery database requires zero 
changes to the agent logic.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseRiskProvider(ABC):
    """
    Abstract Base Class for all risk data providers.
    Defines the contract for retrieving institutional metrics.
    """

    @abstractmethod
    def get_risk_metrics(self, instrument: str, position_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Retrieve risk metrics for a specific instrument and notional.

        Args:
            instrument: Symbol or Asset class (e.g., SONIA, SOFR).
            position_size: Notional amount for scaling sensitivity (DV01).

        Returns:
            Dict containing PV01, DV01, and Compliance status.
        """
        pass


class JSONRiskProvider(BaseRiskProvider):
    """
    Standard implementation using a local JSON data dictionary.
    Optimized for low-latency retrieval during trader demonstrations.
    """

    def __init__(self, data_path: str):
        """
        Initialize the provider with a path to a valid JSON dictionary.

        Args:
            data_path: Absolute or relative path to data_dictionary.json.
        """
        self.data_path = data_path
        self._cache = {}
        self._load_data()

    def _load_data(self):
        """
        Loads metrics from disk into memory. Gracefully handles missing files.
        """
        try:
            with open(self.data_path, "r") as f:
                self._cache = json.load(f)
            logger.info(f"[JSONRiskProvider] Successfully loaded metrics from {self.data_path}")
        except Exception as e:
            logger.error(f"[JSONRiskProvider] Initial load failed: {str(e)}")
            self._cache = {}

    def get_risk_metrics(self, instrument: str, position_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Look up metrics in the local dictionary with support for default fallbacks.
        """
        key = instrument.upper()
        metrics = self._cache.get(key, self._cache.get("DEFAULT", {}))
        
        # Scaling logic for sensitivity metrics (DV01)
        if position_size and "base_dv01" in metrics:
            # Scale DV01 by millions (position_size assumed in units)
            multiplier = position_size / 1_000_000
            metrics["scaled_dv01"] = metrics["base_dv01"] * multiplier
            
        return metrics
