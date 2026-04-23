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
from app.db.operational.service import operational_data_service
from app.services.aidaan.trade_parsing import infer_instrument, parse_notional

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

    @staticmethod
    def _format_limit_summary(limit_row: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Format desk-limit thresholds for trader-facing summaries.
        """
        if not limit_row:
            return None

        currency = limit_row.get("currency") or "USD"
        soft_limit = limit_row.get("soft_limit")
        hard_limit = limit_row.get("hard_limit")

        def _fmt(value: Any) -> str:
            try:
                return f"{float(value):,.0f} {currency}"
            except (TypeError, ValueError):
                return f"{value} {currency}".strip()

        soft_text = _fmt(soft_limit) if soft_limit not in (None, "") else "N/A"
        hard_text = _fmt(hard_limit) if hard_limit not in (None, "") else "N/A"
        return f"{soft_text} / {hard_text}"

    async def _parse_risk_request(
        self,
        *,
        text: str,
        conversation_id: str,
        username: Optional[str],
    ) -> Dict[str, Any]:
        """
        Use Gemini to extract the instrument and requested notional for risk queries.
        """
        prompt = Prompts.RISK_REQUEST_PARSER.format(text=text)
        schema = {
            "type": "object",
            "properties": {
                "instrument": {"type": "string"},
                "position_size": {"type": ["number", "null"]},
                "confidence": {"type": "number"},
            },
            "required": ["instrument"],
        }
        parsed = await self.generate_json_response(
            prompt,
            conversation_id=conversation_id,
            username=username,
            response_schema=schema,
            system_instruction=Prompts.TRADER_SYSTEM_INSTRUCTION + "\nRole: Risk request parser",
        )
        return {
            "instrument": parsed.get("instrument") or "DEFAULT",
            "position_size": parsed.get("position_size"),
            "confidence": parsed.get("confidence"),
        }

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
        
        # 1. Primary path: Gemini extracts the request structure.
        try:
            request_parse = await self._parse_risk_request(
                text=text,
                conversation_id=conversation_id,
                username=context.get("username"),
            )
            instrument = str(request_parse.get("instrument") or "DEFAULT")
            position_size = request_parse.get("position_size")
        except Exception as parse_exc:
            logger.warning("[RiskAgent] Gemini request parser failed; using fallback extraction: %s", parse_exc)
            instrument = infer_instrument(text, default="DEFAULT")
            position_size = parse_notional(text)

        risk_context = operational_data_service.get_risk_context(
            context.get("username"),
            instrument=instrument,
            position_size=position_size,
        )
        
        # 2. Metric retrieval with desk-limit enrichment from PostgreSQL.
        metrics = self.provider.get_risk_metrics(instrument, position_size=position_size)
        selected_limit = risk_context.get("selected_limit")
        if selected_limit:
            metrics["limits"] = {
                "soft": selected_limit.get("soft_limit"),
                "hard": selected_limit.get("hard_limit"),
                "currency": selected_limit.get("currency"),
                "instrument": selected_limit.get("instrument"),
                "source_schema": risk_context.get("source_schema"),
            }
        metrics["requested_notional"] = position_size
        metrics["desk_context"] = {
            "desk": (risk_context.get("desk") or {}).get("name"),
            "title": (risk_context.get("desk_membership") or {}).get("title"),
        }
        metrics["limit_usage"] = risk_context.get("limit_usage", {})
        
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
            dv01 = metrics.get("scaled_dv01", metrics.get("dv01"))
            pv01 = metrics.get("pv01")
            var_99 = metrics.get("var_99")
            limits = metrics.get("limits", {})
            limit_summary = self._format_limit_summary(selected_limit)
            desk_name = (risk_context.get("desk") or {}).get("name")
            requested_notional = metrics.get("requested_notional")
            reply = (
                "[Direct Answer] Risk snapshot assembled from desk metrics.\n"
                "[Context Detail] Use the figures below for pre-trade validation (no auto-execution)."
            )
            bullets = [
                f"Instrument: {instrument}",
                f"Requested Notional: {requested_notional:,.0f}" if requested_notional else "Requested Notional: Not provided",
                f"DV01: {dv01}" if dv01 is not None else "DV01: N/A",
                f"PV01: {pv01}" if pv01 is not None else "PV01: N/A",
                f"VaR(99%): {var_99}" if var_99 is not None else "VaR(99%): N/A",
            ]
            if desk_name:
                bullets.append(f"Desk: {desk_name}")
            if limit_summary:
                bullets.append(f"Desk Limit (Soft/Hard): {limit_summary}")
            elif limits:
                bullets.append(
                    f"Desk Limit (Soft/Hard): {limits.get('soft', 'N/A')} / {limits.get('hard', 'N/A')}"
                )
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
