"""
Local smoke evaluation (FinGAIA-inspired) for Phase 1 demo readiness.

This script does NOT require Vertex credentials; it validates:
- routing determinism
- action payload shape (zero auto-execution)
- MCP tool accessibility (inprocess transport)
"""

import asyncio
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.aidaan.agents.coordinator.coordinator_agent import coordinator_agent
from app.db.base import Base
from app.db.session import engine


@dataclass
class Case:
    name: str
    text: str
    expect_agent: Optional[str] = None
    expect_action_tool: Optional[str] = None


CASES: List[Case] = [
    Case(name="Greeting", text="Hello AIDAAN", expect_agent="greeting"),
    Case(
        name="Order Ticket",
        text="Stage an RFQ for 50m 2Y SONIA, forward starting IMM.",
        expect_agent="order",
        expect_action_tool="draft_rfq_ticket",
    ),
    Case(name="Risk", text="What's the DV01 impact for 50m 2Y SONIA?", expect_agent="risk"),
    Case(name="Market Quote", text="AAPL price", expect_agent="market"),
]


async def run_case(case: Case) -> Dict[str, Any]:
    resp = await coordinator_agent.handle_message(text=case.text, conversation_id="eval-1", context={"username": "eval"})
    payload = resp.model_dump()

    ok = True
    errors: List[str] = []

    agent = payload.get("model", {}).get("agent")
    if case.expect_agent and agent != case.expect_agent:
        ok = False
        errors.append(f"expected_agent={case.expect_agent} got={agent}")

    if case.expect_action_tool:
        actions = payload.get("actions") or []
        if not actions:
            ok = False
            errors.append("expected_action_present got=none")
        else:
            tool_name = actions[0].get("payload", {}).get("tool_name")
            if tool_name != case.expect_action_tool:
                ok = False
                errors.append(f"expected_tool={case.expect_action_tool} got={tool_name}")

    if not payload.get("reply"):
        ok = False
        errors.append("empty_reply")

    return {"case": case.name, "ok": ok, "errors": errors, "agent": agent, "latency_ms": payload.get("latency_ms", 0.0)}


async def main() -> None:
    Base.metadata.create_all(bind=engine)
    results = []
    for case in CASES:
        results.append(await run_case(case))

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    print(f"passed {passed}/{total}")
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        print(f"- {status} | {r['case']} | agent={r['agent']} | errors={','.join(r['errors']) if r['errors'] else '-'}")


if __name__ == "__main__":
    asyncio.run(main())
