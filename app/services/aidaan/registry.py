"""
Agent Registry for AIDAAN.
Centralizes agent discovery and retrieval.
"""
from typing import Dict, Optional
from app.services.aidaan.base import BaseAgent


class AgentRegistry:
    """
    Registry to manage specialized desk agents.
    """
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent_id: str, agent: BaseAgent):
        """
        Registers an agent instance.
        """
        self._agents[agent_id] = agent

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Retrieves an agent by ID.
        """
        return self._agents.get(agent_id)

    def list_agents(self) -> Dict[str, BaseAgent]:
        """
        Returns all registered agents.
        """
        return self._agents


# Global registry instance
registry = AgentRegistry()
