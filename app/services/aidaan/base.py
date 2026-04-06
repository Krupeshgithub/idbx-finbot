"""
Base Agent Abstract Class for AIDAAN.
Provides a standard blueprint for all institutional agents.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from app.schemas.aidaan import ModelInfo, AidaanMessageResponse


class BaseAgent(ABC):
    """
    Abstract base class for all AIDAAN agents.
    Ensures consistency in how agents process messages and interact with tools.
    """

    def __init__(
        self, 
        name: str, 
        model_name: str = "gemini-1.5-pro"
    ):
        self.name = name
        self.model_name = model_name

    @abstractmethod
    async def handle_message(
        self, 
        text: str, 
        conversation_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AidaanMessageResponse:
        """
        Process an incoming message and return a structured AidaanMessageResponse.
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return a list of what this agent can do.
        """
        pass

    def get_model_info(self) -> ModelInfo:
        """
        Return metadata about this agent's underlying model.
        """
        return ModelInfo(
            agent=self.name, 
            llm=self.model_name
        )
