"""
Synthetic seed data for local dev and pre-client demos.
"""
from __future__ import annotations

import logging

from app.core.security import get_password_hash
from app.db.models import (
    AuditEvent,
    Conversation,
    Counterparty,
    Desk,
    DeskLimit,
    DeskMembership,
    Message,
    RFQDraft,
    User,
)
from app.db.session import get_db_session

logger = logging.getLogger(__name__)


def seed_synthetic_data() -> None:
    with get_db_session() as session:
        fx_desk = Desk(
            desk_code="FX_LDN",
            name="London FX Desk",
            location="London",
            asset_class="FX",
            timezone="Europe/London",
            default_context={"default_currencies": ["EUR", "USD", "GBP"]},
        )
        rates_desk = Desk(
            desk_code="RATES_MUM",
            name="Mumbai Rates Desk",
            location="Mumbai",
            asset_class="Rates",
            timezone="Asia/Kolkata",
            default_context={"default_instruments": ["SONIA", "SOFR"]},
        )
        session.add_all([fx_desk, rates_desk])
        session.flush()

        users = [
            User(
                username="trader-001",
                email="trader-001@example.com",
                full_name="Krupesh Patel",
                role="senior_trader",
                hashed_password=get_password_hash("Password123"),
                preferences={"theme": "glass", "voice_mode": True},
            ),
            User(
                username="trader-002",
                email="trader-002@example.com",
                full_name="Priya Shah",
                role="fx_trader",
                hashed_password=get_password_hash("Password123"),
                preferences={"watchlist": ["EUR/USD", "GBP/USD"]},
            ),
            User(
                username="risk-001",
                email="risk-001@example.com",
                full_name="Nisha Rao",
                role="risk_manager",
                hashed_password=get_password_hash("Password123"),
                preferences={"alerts": ["dv01", "var"]},
            ),
        ]
        session.add_all(users)
        session.flush()

        memberships = [
            DeskMembership(user_id=users[0].id, desk_id=fx_desk.id, title="Lead Trader", is_primary=True),
            DeskMembership(user_id=users[1].id, desk_id=fx_desk.id, title="FX Trader", is_primary=True),
            DeskMembership(user_id=users[2].id, desk_id=rates_desk.id, title="Risk Manager", is_primary=True),
        ]
        session.add_all(memberships)

        limits = [
            DeskLimit(desk_id=fx_desk.id, instrument="EUR/USD", limit_type="notional", soft_limit=50_000_000, hard_limit=100_000_000, currency="USD"),
            DeskLimit(desk_id=fx_desk.id, instrument="GBP/USD", limit_type="notional", soft_limit=40_000_000, hard_limit=80_000_000, currency="USD"),
            DeskLimit(desk_id=rates_desk.id, instrument="SONIA", limit_type="dv01", soft_limit=150_000, hard_limit=300_000, currency="GBP"),
        ]
        session.add_all(limits)

        counterparties = [
            Counterparty(desk_id=fx_desk.id, code="BARC", name="Barclays", region="UK", products=["FX Spot", "FX Swap"]),
            Counterparty(desk_id=fx_desk.id, code="HSBC", name="HSBC", region="UK", products=["FX Spot", "NDF"]),
            Counterparty(desk_id=rates_desk.id, code="JPMC", name="JPMorgan", region="US", products=["IRS", "OIS"]),
        ]
        session.add_all(counterparties)

        conversation = Conversation(
            id="conv-seed-001",
            user_id=users[0].id,
            desk_id=fx_desk.id,
            channel="rest",
            title="Morning FX Liquidity Check",
            context_json={"desk_code": fx_desk.desk_code, "source": "synthetic_seed"},
        )
        session.add(conversation)
        session.flush()

        messages = [
            Message(
                conversation_id=conversation.id,
                role="user",
                content="Find liquidity for 25m EUR/USD spot.",
                metadata_json={"source": "synthetic_seed"},
            ),
            Message(
                conversation_id=conversation.id,
                role="assistant",
                content="Top liquidity appears concentrated in London counterparties with manageable spread dispersion.",
                agent_name="market",
                model_name="synthetic",
                metadata_json={"bullets": ["Barclays active", "HSBC active", "Spread normal"]},
            ),
        ]
        session.add_all(messages)

        session.add(
            RFQDraft(
                conversation_id=conversation.id,
                user_id=users[0].id,
                instrument="EUR/USD",
                notional=25_000_000,
                tenor="Spot",
                settlement="T+2",
                payload_json={"direction": "buy", "venue": "IDBX MATCH"},
                requires_human_confirm=True,
            )
        )

        session.add_all(
            [
                AuditEvent(
                    conversation_id=conversation.id,
                    event_type="seed_initialized",
                    severity="info",
                    actor="system",
                    summary="Synthetic desk and trader context loaded.",
                    payload_json={"desk": fx_desk.desk_code},
                ),
                AuditEvent(
                    conversation_id=conversation.id,
                    event_type="rfq_draft_created",
                    severity="info",
                    actor="system",
                    summary="Synthetic RFQ draft prepared for EUR/USD spot.",
                    payload_json={"instrument": "EUR/USD", "notional": 25_000_000},
                ),
            ]
        )

    logger.info("[Database] Synthetic seed data loaded.")

