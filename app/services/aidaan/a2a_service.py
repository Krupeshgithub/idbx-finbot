"""
A2A (Agent-to-Agent) Service for AIDAAN.
Facilitates communication between the Coordinator and specialized agents.
"""
from typing import Any, Dict, Optional
from app.services.aidaan.risk_agent import risk_agent
from app.services.aidaan.distributor_agent import distributor_agent


class A2AService:
    """
    Standardizes how agents request information from one another.
    """

    async def query_agent(
        self, 
        target_agent: str, 
        message: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route a query to the appropriate agent and return their response.
        """
        if target_agent == "risk":
            return await risk_agent.handle_message(message, "internal-a2a", context)
        elif target_agent == "distributor":
            return await distributor_agent.handle_message(message, "internal-a2a", context)
        
        return {"error": f"Agent '{target_agent}' not found or unreachable via A2A."}


a2a_service = A2AService()
