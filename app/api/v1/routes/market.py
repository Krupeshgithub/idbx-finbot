"""
AIDANN Market data routes.
"""
from fastapi import APIRouter, Query
from app.services.aidann.lseg_service import lseg_service
from app.services.aidann.bq_service import bq_service


router = APIRouter()


@router.get("/snapshot/{instrument}")
async def snapshot(instrument: str):
    """
    This endpoint allows users to get a market snapshot for a specific instrument.
    """
    return await lseg_service.get_snapshot(instrument)


@router.get("/history/{instrument}")
async def history(
    instrument: str,
    start: str = Query(..., description="Inclusive ISO-8601 start timestamp"),
    end: str = Query(..., description="Inclusive ISO-8601 end timestamp")
):
    """
    Returns market history from LDL/BigQuery for the given instrument and time range.
    """
    return await bq_service.get_historical_ticks(instrument, start, end)
