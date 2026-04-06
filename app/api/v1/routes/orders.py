"""
Order routes for AIDANN.
"""
from fastapi import APIRouter
from app.utils.financial import draft_rfq_ticket


router = APIRouter()


@router.post("/rfq")
async def rfq(
    instrument: str,
    notional: float
):
    """
    This endpoint allows users to draft a Request for Quote (RFQ) ticket for a specific instrument and notional amount.
    """
    return draft_rfq_ticket(instrument, notional, ["DEALER1"])
