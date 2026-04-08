"""
Base Agent Abstract Class for AIDAAN.
Provides a standard blueprint for all institutional agents.

CHANGES FROM ORIGINAL:
- Removed direct `genai.configure()` and `GenerativeModel(...)` calls.
- All LLM calls now go through `llm_client` (app.core.llm_client).
- `self.model` is still available for any agent that needs direct SDK access,
    but prefer `self.llm` (the shared LLMClient) for new code.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.core.llm_client import llm_client
from app.schemas.aidaan import ModelInfo, AidaanMessageResponse
from app.core.config.settings import settings


class BaseAgent(ABC):
    """
    Abstract base class for all AIDAAN agents.
    """

    def __init__(
        self,
        name: str,
        model_name: str = settings.VERTEX_AI_MODEL_NAME
    ):
        self.name = name
        self.model_name = model_name
        self.llm = llm_client
        self.model = llm_client._get_model(model_name)

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
