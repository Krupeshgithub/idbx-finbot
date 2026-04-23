"""
Institutional market agent for Vertex-first market orchestration.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.config.settings import settings
from app.core.prompts import Prompts
from app.schemas.aidaan import AidaanMessageResponse
from app.services.aidaan.core.base import BaseAgent

logger = logging.getLogger(__name__)


class MarketAgent(BaseAgent):
    """
    Vertex-first market agent.

    This agent intentionally avoids manual ticker aliases and heuristic
    fast-paths so symbol resolution, tool choice, and response synthesis
    remain model-driven.
    """

    def __init__(self) -> None:
        """
        Initialize the market agent with the default market model.
        """
        super().__init__(name="market", model_name=settings.VERTEX_AI_MODEL_NAME)

    async def handle_message(
        self,
        text: str,
        conversation_id: str,
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
    ) -> AidaanMessageResponse:
        """
        Run a full Vertex-driven market workflow for the user query.

        The model is responsible for:
        - identifying the correct instrument or company
        - choosing the relevant MCP tools
        - synthesizing the final trader-facing answer
        """
        logger.info("[MarketAgent] Analyzing market query via Vertex-first orchestration: %s", text)
        context = context or {}

        orchestration_prompt = Prompts.MARKET_ORCHESTRATION.format(text=text)
        model_name = settings.VERTEX_AI_REASONING_MODEL_NAME
        logger.info("[MarketAgent] Routing to reasoning model: %s", model_name)

        try:
            response_data = await self.generate_json_response(
                orchestration_prompt,
                conversation_id=conversation_id,
                model_override=model_name,
                username=context.get("username"),
                use_mcp_tools=True,
                tool_callback=tool_callback,
                response_schema=Prompts.STANDARD_RESPONSE_SCHEMA,
                system_instruction=(
                    Prompts.TRADER_SYSTEM_INSTRUCTION
                    + "\nRole: Senior Interbank analyst."
                    + " Resolve the correct ticker or instrument from the user's wording before using tools."
                    + " Use MCP tools to validate the symbol and gather only the data required for the question."
                    + " For simple quote requests, keep the answer concise and accurate."
                    + " For historical requests, prefer historical series tools over spot quote tools."
                ),
            )
            return self.build_message_response(
                reply=response_data.get("reply", "Market data analysis complete."),
                bullets=response_data.get("bullets", []),
                conversation_id=conversation_id,
                model_info=self.get_model_info(model_override=model_name),
            )
        except Exception as exc:
            return self.build_error_response(
                reply="I encountered an issue while performing the market analysis. Please verify the instrument or request.",
                conversation_id=conversation_id,
                error=exc,
                model_info=self.get_model_info(model_override=model_name),
            )

    def get_capabilities(self) -> List[str]:
        """
        Return the market coverage supported by the agent.
        """
        return [
            "market_quotes",
            "historical_price_series",
            "technical_analysis",
            "fundamental_analysis",
            "multi_tool_market_orchestration",
        ]


market_agent = MarketAgent()
