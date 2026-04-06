"""
Distributor Agent Implementation for AIDAAN.
Institutional trade distribution skeleton.
"""
from typing import Any, Dict, List, Optional
from app.services.aidaan.base import BaseAgent
from app.schemas.aidaan import ActionItem, AidaanMessageResponse


class DistributorAgent(BaseAgent):
    """
    Handles distribution and matching logic.
    """

    def __init__(self):
        super().__init__(
            name="distributor_agent", 
            model_name="gemini-1.5-pro"
        )

    async def handle_message(
        self, 
        text: str, 
        conversation_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AidaanMessageResponse:
        """
        Identify liquidity providers and match logic.
        """
        lowered = text.lower()
        if "find" in lowered or "search" in lowered:
            return AidaanMessageResponse(
                reply="I'm scanning the pool for deep liquidity matches...",
                bullets=["Pool: IDBX-Internal", "Depth: $500M+", "Status: SCANNING"],
                actions=[],
                conversation_id=conversation_id,
                model=self.get_model_info()
            )

        return AidaanMessageResponse(
            reply="Liquidity distribution nodes are optimized for your current desk location (EMEA).",
            bullets=[
                "Matches Found: IDBX-POOL, LSEG-MATCH",
                "Liquidity Status: READY",
                "Distribution Node: EMEA-HUB-01"
            ],
            actions=[
                ActionItem(
                    kind="open_view",
                    title="View Matching Grid",
                    payload={"view": "matching_providers"},
                    requires_human_confirm=False
                ),
                ActionItem(
                    kind="distribute_now",
                    title="Execute Distribution",
                    payload={"node": "EMEA-HUB-01"},
                    requires_human_confirm=True
                )
            ],
            conversation_id=conversation_id,
            model=self.get_model_info()
        )

    def get_capabilities(self) -> List[str]:
        return ["liquidity_discovery", "rfq_distribution", "match_optimization"]


distributor_agent = DistributorAgent()
