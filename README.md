# AIDAAN — AI-Powered Interbank Assistant

**AIDAAN** is an intelligent, always-present conversational assistant designed for the **IDBX Live Trading Platform**. It functions as a "one-piece phenomenon" on the trader's desk, providing real-time market intelligence, RFQ drafting, and desk-to-desk collaboration via institutional-grade protocols.

---

## Architecture: Why REST + WebSocket?

In a high-stakes trading environment, transparency and reliability are paramount. We use a **Dual-Channel Architecture**:

1.  **REST API (Stateless/Auditable)**: Used for formal "conversational turns" (e.g., `POST /v1/aidaan/message`). This ensures every trade-intent message is logged for compliance and auditing.
2.  **WebSocket (Real-time/Stateful)**: Used for the "Glass Sphere" visual state machine (Listening, Thinking, Alerts) and low-latency interactive chat. This provides the "always-alive" feeling without polling the server.

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- A virtual environment (recommended)

### 2. Installation
```bash
# Register the virtual environment
source .venv/bin/activate  # On Linux/macOS

# Install institutional dependencies
pip install -r requirements.txt
```

### 3. Vertex AI Configuration
Create a `.env` file with the required runtime configuration:

```env
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=global
VERTEX_AI_MODEL_NAME=gemini-2.5-flash
VERTEX_AI_REASONING_MODEL_NAME=gemini-3.1-pro-preview
VERTEX_AI_ROUTER_MODEL_NAME=gemini-3.1-flash
VERTEX_AI_API_VERSION=v1
VERTEX_AI_USE_EXPRESS_MODE=true
VERTEX_AI_SERVICE_ACCOUNT_FILE=/absolute/path/to/service-account.json
VERTEX_AI_SERVICE_ACCOUNT_JSON=
VERTEX_AI_ENABLE_GOOGLE_SEARCH=false
VERTEX_AI_ENABLE_URL_CONTEXT=false
VERTEX_AI_ENABLE_CODE_EXECUTION=false
DATABASE_URL=sqlite:///./aidann.db
MCP_TRANSPORT=inprocess
```

Notes:
- Authentication is now Vertex AI based. The app uses Application Default Credentials by default.
- If you don't have service-account credentials, the app can try Vertex AI Express Mode with `GOOGLE_API_KEY`.
- `VERTEX_AI_SERVICE_ACCOUNT_FILE` is optional and only needed when you do not want to rely on ADC.
- `VERTEX_AI_SERVICE_ACCOUNT_JSON` is useful for Docker when you want to inject the full service-account JSON via env instead of mounting a file.
- `gemini-2.5-flash` remains a strong default model, `gemini-3.1-pro-preview` is configured for heavier reasoning flows, and `gemini-3.1-flash-lite-preview` is used for low-latency routing.
- Vertex server tools such as Google Search / URL Context / Code Execution are supported as optional toggles and stay off by default for predictable latency.

### 4. Running the Server
```bash
uvicorn asgi:app --reload
```
The server will start at `http://localhost:8000`.

Useful dev checks:
- `GET /health`
- `GET /v1/diagnostics/config` (Vertex + MCP + cache status)

Local smoke evaluation (no Vertex required):
```bash
DATABASE_URL=sqlite:///./aidann.db .venv/bin/python app/eval/smoke_eval.py
```

### 5. Docker
```bash
cp .env.example .env
docker compose up --build
```

If Vertex credentials are not configured yet, the container will still boot and `/health` will work, but LLM-backed routes will return a clear configuration error until you add the required Vertex settings.

---

## Detailed Postman Testing Guide

Testing AIDAAN requires a specific flow because of the **JWT Security Layer**. Follow these steps:

### Step 1: Institutional Login
AIDAAN requires a secure token to interact with the desk agents.
- **Method**: `POST`
- **URL**: `http://localhost:8000/v1/auth/login`
- **Body** (JSON):
  ```json
  {
    "username": "trader-001",
    "password": "password123"
  }
  ```
- **Action**: Click **Send**. Copy the `access_token` from the response.

### Step 2: Configure Authorization (The Key Step)
For all other requests (except login), you must provide the token:
1. In Postman, go to the **Auth** tab.
2. Select **Type**: `Bearer Token`.
3. Paste the token you copied from Step 1 into the **Token** field.

### Step 3: Send a Message (REST)
- **Method**: `POST`
- **URL**: `http://localhost:8000/v1/aidaan/message`
- **Body** (JSON):
  ```json
  {
    "user_id": "trader-001",
    "text": "Draft an RFQ for 50 million EUR/PLN Swap 3M",
    "language": "en"
  }
  ```
- **Action**: Click **Send**. You will receive a structured response with `bullets` and `actions`.

### Step 4: Test the WebSocket (Real-time)
Postman now supports WebSocket testing!
1. Click **New** -> **WebSocket Request**.
2. **URL**: `ws://localhost:8000/v1/ws/aidaan`
3. Click **Connect**.
4. In the **Message** box, type:
   ```json
   {
     "user_id": "trader-001",
     "text": "Check liquidity for Swap 3M"
   }
   ```
5. Click **Send**. You will see the visual state transitions (`Listening` -> `Thinking` -> `Reply`) in the log.

---

## Compliance Guardrails (Hardcoded)
- **Zero Auto-Execution**: Every tool action (like `draft_rfq`) returns a `requires_human_confirm: true` flag. No trade can be executed without manual confirmation.
- **Non-Advisory Filter**: Asking "Should I buy?" will trigger a compliance block, reframing the response as a factual recap.

---

## Interactive Testing Frontend

To see AIDAAN in action with the **Glass Sphere** visual states, we have included a standalone test interface:

1.  Open `test_frontend.html` in any modern web browser (Chrome, Edge, Firefox).
2.  Ensure the backend is running (`uvicorn asgi:app --reload`).
3.  Log in using the default credentials:
    - **Username**: `trader-001`
    - **Password**: `password123`
4.  Interact with the sphere! Try phrases like:
    - *"Draft an RFQ for 50mm EUR/PLN"*
    - *"How can I fund 5bn overnight?"*
    - *"Should I buy EUR/USD?"* (Tests compliance guardrails)
