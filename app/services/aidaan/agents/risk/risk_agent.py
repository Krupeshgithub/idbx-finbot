"""
Institutional Risk Analyst Agent - AIDAAN Contextual Guardrail
=============================================================
The RiskAgent evaluates trade parameters, desk limits, and portfolio 
sensitivities using a pluggable RAG (Retrieval-Augmented Generation) 
service.

Key Features:
- Contextual risk metric retrieval (DV01, PV01, VaR).
- Personalized desk limit validation.
- Agentic synthesis of complex risk reports.
"""

import logging
from typing import Any, Dict, List, Optional

from app.services.aidaan.core.base import BaseAgent
from app.schemas.aidaan import AidaanMessageResponse
from app.core.prompts import Prompts
from app.services.aidaan.providers.risk_provider import JSONRiskProvider
from app.core.config.settings import settings

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    """
    Evaluates institutional risk metrics and provides synthesized 
    compliance summaries for trading desks.
    """

    def __init__(self, provider: JSONRiskProvider):
        """
        Initialize with a data provider for risk metrics.

        Args:
            provider: Pluggable data source (JSON/BigQuery) for risk data.
        """
        super().__init__(name="risk", model_name=settings.VERTEX_AI_MODEL_NAME)
        self.provider = provider

    async def handle_message(
        self, 
        text: str, 
        conversation_id: str, 
        context: Optional[Dict[str, Any]] = None,
        tool_callback: Optional[callable] = None,
    ) -> AidaanMessageResponse:
        """
        Processes risk-related queries by fetching context from the RAG provider.

        Args:
            text: Risk query (e.g., "What is the DV01 impact of a 50m SONIA swap?").
            conversation_id: Session identifier.
            context: Metadata for personalized risk evaluation.

        Returns:
            AidaanMessageResponse: A detailed risk evaluation report.
        """
        logger.info(f"[RiskAgent] Analyzing risk query: {text}")
        context = context or {}
        
        # 1. Subject Extraction (Heuristic mapping for demo/showcase efficiency)
        instrument = "DEFAULT"
        lowered = text.lower()
        if "sonia" in lowered: instrument = "SONIA"
        elif "sofr" in lowered: instrument = "SOFR"
        elif "cable" in lowered or "gbp/usd" in lowered: instrument = "GBP/USD"
        elif "gilt" in lowered: instrument = "UK GILTS"
        
        # 2. RAG Retrieval phase
        # Position size extracted/assumed for demo purposes
        metrics = self.provider.get_risk_metrics(instrument, position_size=50_000_000)
        
        # 3. Final synthesis using externalized prompt
        prompt = Prompts.RISK_SYNTHESIS.format(
            text=text,
            instrument=instrument,
            metrics=metrics
        )
        
        try:
            parsed = await self.generate_json_response(
                prompt,
                conversation_id=conversation_id,
                username=context.get("username"),
            )
            return self.build_message_response(
                reply=parsed.get("reply", "Risk analysis completed."),
                bullets=parsed.get("bullets", []),
                conversation_id=conversation_id,
                model_info=self.get_model_info(),
            )
        except Exception as e:
            logger.warning("[RiskAgent] LLM synthesis failed; using deterministic formatting: %s", e)
            dv01 = metrics.get("dv01")
            pv01 = metrics.get("pv01")
            var_99 = metrics.get("var_99")
            limits = metrics.get("limits", {})
            reply = (
                "[Direct Answer] Risk snapshot assembled from desk metrics.\n"
                "[Context Detail] Use the figures below for pre-trade validation (no auto-execution)."
            )
            bullets = [
                f"Instrument: {instrument}",
                f"DV01: {dv01}" if dv01 is not None else "DV01: N/A",
                f"PV01: {pv01}" if pv01 is not None else "PV01: N/A",
                f"VaR(99%): {var_99}" if var_99 is not None else "VaR(99%): N/A",
                f"Limit Used: {limits.get('used', 'N/A')} / {limits.get('max', 'N/A')}",
            ]
            return self.build_message_response(
                reply=reply,
                bullets=bullets,
                conversation_id=conversation_id,
                model_info={"agent": self.name, "llm": "fallback-template"},
            )

    def get_capabilities(self) -> List[str]:
        """
        Lists specific risk analysis domains handled by this agent.
        """
        return ["contextual_risk_rag", "dv01_analysis", "pv01_calculation", "limit_validation"]


# Professional initialization with local provider
# This is architected for a 1-line swap to BigQueryRiskProvider in the future.
risk_provider = JSONRiskProvider("app/services/aidaan/providers/data_dictionary.json")
risk_agent = RiskAgent(risk_provider)
