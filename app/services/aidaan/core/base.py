"""
Base Agent and Registry for AIDAAN
===================================
Common interface for all specialized agents.
"""
import logging
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from app.core.llm_client import llm_client

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract Base Class for all specialized agents in AIDAAN.
    Provides a unified interface for message handling, tool usage, 
    and model metadata reporting.
    """
    def __init__(self, name: str, model_name: Optional[str] = None):
        """
        Initialize the agent with a name and a designated LLM.

        Args:
            name: Human-readable ID for the agent (e.g., 'market').
            model_name: Optional override for the underlying Gemini model.
        """
        self.name = name
        self.llm = llm_client
        self._model_name = model_name

    @abstractmethod
    async def handle_message(
        self, 
        text: str, 
        conversation_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Asynchronously processes a user message.

        Args:
            text: Input string from the user.
            conversation_id: Session ID for context.
            context: Additional metadata.

        Returns:
            Varies by agent (usually AidaanMessageResponse).
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Returns a list of high-level features provided by this agent.
        """
        pass

    def get_model_info(self) -> Dict[str, str]:
        return {
            "agent": self.name,
            "llm": self._model_name or "gemini-1.5-pro"
        }


class AgentRegistry:
    """
    Central registry for managing the lifecycle and discovery 
    of specialized AI agents.
    """
    def __init__(self):
        """
        Initializes an empty agent repository.
        """
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent_id: str, agent: BaseAgent):
        """
        Binds an agent instance to a unique ID.

        Args:
            agent_id: Primary key for agent lookup (e.g., 'risk').
            agent: The BaseAgent implementation instance.
        """
        self._agents[agent_id] = agent
        logger.info(f"[Registry] Registered agent: {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Retrieves a registered agent by its ID.
        """
        return self._agents.get(agent_id)

    def list_agents(self) -> List[str]:
        """
        Returns a list of all currently registered agent IDs.
        """
        return list(self._agents.keys())

registry = AgentRegistry()
