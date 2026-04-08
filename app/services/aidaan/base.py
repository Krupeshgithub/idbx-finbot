"""
Base Agent Abstract Class for AIDAAN.
Provides a standard blueprint for all institutional agents.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from app.schemas.aidaan import ModelInfo, AidaanMessageResponse
from app.core.config.settings import settings


class BaseAgent(ABC):
    """
    Abstract base class for all AIDAAN agents.
    Ensures consistency in how agents process messages and interact with tools.
    """

    def __init__(
        self, 
        name: str, 
        model_name: str = settings.VERTEX_AI_MODEL_NAME
    ):
        self.name = name
        self.model_name = model_name

        genai.configure(api_key=settings.GOOGLE_API_KEY)

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        )

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
