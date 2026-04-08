"""
Institutional Guardrails for AIDAAN
Hardcoded safety constraints for interbank compliance.

CHANGES FROM ORIGINAL:
- `get_compliance_notice()` now returns Prompts.COMPLIANCE_NOTICE
    so the text lives in one place only.
"""
from typing import Tuple
from app.core.prompts import Prompts


class Guardrails:
    """
    Enforces non-negotiable compliance rules.
    """

    @staticmethod
    def is_advisory(text: str) -> Tuple[bool, str]:
        """
        Detects if the user is asking for financial advice.
        Returns (is_advisory, reason).

        To add new keywords, edit this list — or move to settings.py
        if they need to be environment-configurable.
        """
        advisory_keywords = [
            "should i buy",
            "should i sell",
            "is it a good time",
            "recommendation"
        ]
        lowered_text = text.lower()

        for kw in advisory_keywords:
            if kw in lowered_text:
                return True, f"Detected advice-seeking query via keyword: '{kw}'"

        return False, ""

    @staticmethod
    def get_compliance_notice() -> str:
        """
        Standard disclaimer — text lives in Prompts.COMPLIANCE_NOTICE.
        """
        return Prompts.COMPLIANCE_NOTICE


guardrails = Guardrails()
