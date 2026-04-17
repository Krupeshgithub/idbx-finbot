"""
Base Agent and Registry for AIDAAN
===================================
Common interface for all specialized agents.
"""
import logging
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from app.core.llm_client import llm_client
from app.core.config.settings import settings
from app.schemas.aidaan import AidaanMessageResponse
from app.services.aidaan.runtime_context import runtime_context_service

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
            model_name: Optional override for the underlying Vertex AI model.
        """
        self.name = name
        self.llm = llm_client
        self._model_name = model_name

    @abstractmethod
    async def handle_message(
        self, 
        text: str, 
        conversation_id: str, 
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
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

    def get_model_info(self, model_override: Optional[str] = None) -> Dict[str, str]:
        return {
            "agent": self.name,
            "llm": model_override or self._model_name or settings.VERTEX_AI_MODEL_NAME
        }

    async def generate_json_response(
        self,
        prompt: str,
        *,
        conversation_id: Optional[str] = None,
        model_override: Optional[str] = None,
        username: Optional[str] = None,
        use_mcp_tools: bool = False,
        tool_callback: Optional[callable] = None,
        response_schema: Optional[Any] = None,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Shared helper to call the LLM and normalize model-side error payloads.
        """
        if conversation_id:
            prompt = runtime_context_service.build_prompt_context(
                base_prompt=prompt,
                conversation_id=conversation_id,
                username=username,
            )
        parsed = await self.llm.generate_json(
            prompt=prompt,
            model_override=model_override or self._model_name,
            use_mcp_tools=use_mcp_tools,
            tool_callback=tool_callback,
            response_schema=response_schema,
            system_instruction=system_instruction,
        )
        if parsed.get("error"):
            raise RuntimeError(parsed["error"])
        return parsed

    def build_message_response(
        self,
        *,
        reply: str,
        conversation_id: str,
        bullets: Optional[List[str]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        model_info: Optional[Dict[str, str]] = None,
        latency_ms: float = 0.0,
    ) -> AidaanMessageResponse:
        """
        Create a standardized message response for all agents.
        """
        return AidaanMessageResponse(
            reply=reply,
            bullets=bullets or [],
            actions=actions or [],
            conversation_id=conversation_id,
            model=model_info or self.get_model_info(),
            latency_ms=latency_ms,
        )

    def build_error_response(
        self,
        *,
        reply: str,
        conversation_id: str,
        error: Exception | str,
        bullets: Optional[List[str]] = None,
        model_info: Optional[Dict[str, str]] = None,
    ) -> AidaanMessageResponse:
        """
        Create a standardized fallback response while preserving the underlying error.
        """
        error_text = str(error)
        fallback_bullets = list(bullets or [])
        if not fallback_bullets:
            fallback_bullets = [error_text]

        logger.error("[%s] Request failed: %s", self.__class__.__name__, error_text)
        return self.build_message_response(
            reply=reply,
            bullets=fallback_bullets,
            actions=[],
            conversation_id=conversation_id,
            model_info=model_info or self.get_model_info(),
        )


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
