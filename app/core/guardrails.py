"""
Institutional Guardrails for AIDAAN.
Hardcoded safety constraints for interbank compliance.
"""
from typing import List, Tuple


class Guardrails:
    """
    Enforces non-negotiable compliance rules.
    """

    @staticmethod
    def is_advisory(text: str) -> Tuple[bool, str]:
        """
        Detects if the user is asking for financial advice.
        Returns (is_advisory, reason).
        """
        advisory_keywords = ["should i buy", "should i sell", "is it a good time", "recommendation"]
        lowered_text = text.lower()
        
        for kw in advisory_keywords:
            if kw in lowered_text:
                return True, f"Detected advice-seeking query via keyword: '{kw}'"
        
        return False, ""

    @staticmethod
    def get_compliance_notice() -> str:
        """
        Standard disclaimer for non-advisory responses.
        """
        return "I am a factual assistant and cannot provide financial advice or recommendations. My responses are based on market data and system state."


guardrails = Guardrails()
