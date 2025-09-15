# Reine des Maracas – Root/SAV/Sales Agents

This project wires:

* **MCP tools** (`orders.get_status`, `returns.create_label`, `catalog.search`, `stores.hours`)
* **RAG** for returns/refund policy (Agentspace / BigQuery) + local Chroma fallback
* **Google Maps** for live store search
* **ADK** agents (Root → routes to SAV/Sales) with CLI & Web UI
* **(Optional)** deploy to **Vertex AI Agent Engine**

---

## 1) Environment variables

### Must-have (used by agents & tools)

* `MCP_HTTP_URL` – MCP HTTP gateway base URL (e.g., `http://localhost:8080` or Cloud Run URL)
* `MCP_HTTP_INVOKE_PATH` – invoke path (default `/tools/invoke`)
* `AGENTSPACE_URL` – Agentspace UI URL (optional; EU default below)

  * Default: `https://vertexaisearch.cloud.google.com/eu/home/cid/6a59ad39-5ee7-41c7-ae22-021eb8cc1998?hl=en_GB`
* `VERTEX_PROJECT` – GCP project **for BigQuery** (default `avisia-training`)
* `VERTEX_BQ_DATASET` – BigQuery dataset (default `reine_des_maracas`)
* `VERTEX_LOCATION` – Region (default `europe-west1`)
* `GOOGLE_MAPS_API_KEY` – required for live store search (otherwise mock stores)
* `CHROMA_PATH` (optional) – local Chroma vector DB path
* `CHROMA_COLLECTION` (optional) – default `reine_des_maracas_policies`

### Must-have (for **ADK + Vertex models**)

> ADK’s Google GenAI client expects the **`VERTEXAI_*`** names.

* `GOOGLE_GENAI_USE_VERTEXAI=true` (or `GENAI_VERTEXAI=true`)
* `VERTEXAI_PROJECT=avisia-training`
* `VERTEXAI_LOCATION=europe-west1`
* (recommended) `GOOGLE_CLOUD_PROJECT=$VERTEXAI_PROJECT`
* (recommended) `GOOGLE_CLOUD_LOCATION=$VERTEXAI_LOCATION`

> If your `.env` uses `VERTEX_PROJECT`/`VERTEX_LOCATION`, **map them**:

```
VERTEXAI_PROJECT=${VERTEX_PROJECT}
VERTEXAI_LOCATION=${VERTEX_LOCATION}
```

---

## 2) MCP tools expected

Gateway must expose **both** canonical & dotted names (we use dotted in code):

* `agentspace.search` – RAG over Agentspace / Vertex AI Search (optionally BigQuery tables)
* `orders.get_status` – order status
* `returns.create_label` – create return label
* `catalog.search` – catalog search
* `stores.hours` – store hours

### Call shape (from this app)

**POST** body:

```json
{ "tool": "<name>", "arguments": { ... } }
```

### Response shapes accepted

* Raw dict/list, **or**
* Wrapped: `{ "result": { ... } }`

---

## 3) RAG backend priority

1. **Agentspace (Vertex AI Search)** over BigQuery:

   * Project: `avisia-training`
   * Dataset: `reine_des_maracas`
   * Region: `europe-west1`

2. **Chroma DB** if `CHROMA_PATH` set and `chromadb` is installed

3. **In-memory** mini-KB (Jaccard) fallback

---

## 4) Project files

```
.
├─ agent.py                # Root/SAV/Sales agents, MCP tools, routing, guardrail
├─ cli.py                  # Handy CLI commands (policy/order/label/catalog/stores/chat)
├─ mcp_server.py           # Minimal FastAPI MCP HTTP gateway (tools.*)
├─ requirements.txt
├─ adk_app.py              # One-liner for ADK Web UI:  from agent import root_agent as AGENT
├─ apps/                   # (optional) ADK Apps folder if you add more apps
└─ .env                    # Fill me (see template below)
```

---

## 5) .env template

```dotenv
# === MCP gateway ===
MCP_HTTP_URL=http://localhost:8080
# MCP_HTTP_INVOKE_PATH=/tools/invoke
# MCP_HTTP_BEARER=           # optional bearer auth

# === Agentspace / RAG ===
AGENTSPACE_URL=https://vertexaisearch.cloud.google.com/eu/home/cid/6a59ad39-5ee7-41c7-ae22-021eb8cc1998?hl=en_GB
VERTEX_PROJECT=avisia-training
VERTEX_BQ_DATASET=reine_des_maracas
VERTEX_LOCATION=europe-west1
# VERTEX_BQ_TABLES=complement_individu,individu,magasin,referentiel,ticket_caisse,typo_produit

# === Vertex GenAI (ADK models) ===
GOOGLE_GENAI_USE_VERTEXAI=true
VERTEXAI_PROJECT=${VERTEX_PROJECT}
VERTEXAI_LOCATION=${VERTEX_LOCATION}
# (optional but useful)
GOOGLE_CLOUD_PROJECT=${VERTEX_PROJECT}
GOOGLE_CLOUD_LOCATION=${VERTEX_LOCATION}

# === Google Maps ===
GOOGLE_MAPS_API_KEY=

# === Chroma (optional) ===
# CHROMA_PATH=./.chroma
# CHROMA_COLLECTION=reine_des_maracas_policies
```

---

## 6) Install & quick local tests

```bash
# Use your venv
uv pip install -r requirements.txt
# Or:
pip install -r requirements.txt
```

**Model note**: we pin to `gemini-1.5-flash-002`.
If you previously used `gemini-2.0-flash-exp`, switch to `gemini-1.5-flash-002` (the exp model may not exist in your region).

### CLI smoke tests

Terminal A – **MCP gateway**:

```bash
uvicorn mcp_server:app --host 0.0.0.0 --port 8080
```

Terminal B – **agent CLI** (same shell must have env vars):

```bash
export MCP_HTTP_URL="http://localhost:8080"
# export MCP_HTTP_INVOKE_PATH="/tools/invoke"   # default already ok

# Vertex creds (same shell):
gcloud auth application-default login
gcloud auth application-default set-quota-project avisia-training
export GOOGLE_GENAI_USE_VERTEXAI=true
export VERTEXAI_PROJECT=avisia-training
export VERTEXAI_LOCATION=europe-west1
export GOOGLE_CLOUD_PROJECT=$VERTEXAI_PROJECT
export GOOGLE_CLOUD_LOCATION=$VERTEXAI_LOCATION

# Optional: live stores
export GOOGLE_MAPS_API_KEY=YOUR_KEY

# RAG
python3 cli.py rag "Puis-je retourner un article après 30 jours ?"

# Orders
python3 cli.py order-status 123456

# Returns
python3 cli.py return-label 123456 "Taille incorrecte"

# Catalog
python3 cli.py search-products "bikini noir" --size M --color noir --limit 3

# Stores
python3 cli.py find-store -q "Paris"
python3 cli.py store-hours "gmaps:opera-123"

# Chat (Root agent)
python3 cli.py chat --ask "Je veux retourner un maillot"
```

---

## 7) ADK Web UI (built-in)

We expose the root agent via `adk_app.py` (required by ADK Web):

```python
# adk_app.py
from agent import root_agent as AGENT
```

Run the web UI from **project root**:

```bash
export PYTHONPATH="$PWD"

# Ensure Vertex env is set in THIS shell (see section 6)
export GOOGLE_GENAI_USE_VERTEXAI=true
export VERTEXAI_PROJECT=avisia-training
export VERTEXAI_LOCATION=europe-west1

# Launch the web UI that auto-loads AGENT from adk_app.py
adk web apps
# Open http://127.0.0.1:8000
```

> If you see “Missing key inputs argument…”, your shell running `adk web` doesn’t have `VERTEXAI_*` exports. Re-export and relaunch.

---

## 8) Deploy the MCP HTTP Gateway (Docker + Cloud Run)

**Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Build & push to Artifact Registry**

```bash
gcloud config set project avisia-training
gcloud artifacts repositories create mcp-gateway \
  --repository-format=DOCKER --location=europe-west1 || true

# IMPORTANT: run from the folder **that contains Dockerfile**
gcloud builds submit --tag \
  europe-west1-docker.pkg.dev/avisia-training/mcp-gateway/mcp-http:v1
```

**Deploy to Cloud Run**

```bash
gcloud run deploy mcp-http \
  --image europe-west1-docker.pkg.dev/avisia-training/mcp-gateway/mcp-http:v1 \
  --region europe-west1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --set-env-vars "PYTHONUNBUFFERED=1"
```

**Sanity checks**

```bash
# list tools
curl -s https://<your-cloud-run-url>/tools | jq

# invoke (dotted name + correct arg names)
curl -s -X POST https://<your-cloud-run-url>/tools/invoke \
  -H 'Content-Type: application/json' \
  -d '{"tool":"orders.get_status","arguments":{"id":"123"}}' | jq
```

> If you get `jq: parse error`, your shell echo includes extra lines. Copy only the `curl` lines.

Set `MCP_HTTP_URL` to your Cloud Run URL for the agents.

---

## 9) Deploy to Vertex AI Agent Engine (optional)

We already validated local behavior; next, package & deploy.

**Prereqs**

```bash
gcloud auth application-default login
gcloud config set project avisia-training

export GOOGLE_GENAI_USE_VERTEXAI=true
export VERTEXAI_PROJECT=avisia-training
export VERTEXAI_LOCATION=europe-west1
```

**Deploy (using your helper script / flow)**
You likely used a function that:

* Pickles the agent runner to `gs://<proj>-adk-staging/agent_engine/agent_engine.pkl`
* Uploads `requirements.txt` (auto-adds `cloudpickle`)
* Creates the Reasoning Engine

Watch the output for a **resource name** like:

```
projects/1062626335546/locations/europe-west1/reasoningEngines/5986145921891565568
```

If the LRO fails to start:

* Open logs from the link printed (Cloud Logging)
* Common fixes:

  * Ensure model exists in region (`gemini-1.5-flash-002`)
  * Add missing deps (tooling libs) to `requirements.txt`
  * Remove ephemeral test packages pulled in by your venv

Once the Reasoning Engine is **RUNNING**, you can use the **Vertex “Web app” integration** to chat with it, or call it from your own UI/API.

---

## 10) Agentspace wiring

If you want Agentspace to route to Agent Engine:

* Open **Vertex AI → Integrations → Web app** (or your internal Agentspace admin)
* Add a **Vertex AI Agent Engine** source
* Paste the **resource name**:

```
projects/1062626335546/locations/europe-west1/reasoningEngines/5986145921891565568
```

* Save. Messages are now routed to your agent.

> If you don’t see an “Agentspace admin” in your environment, using Vertex’s “Integrations → Web app” is the simplest official UI to test.

---

## 11) Troubleshooting cheatsheet

* **`Missing key inputs argument!`**
  Ensure the shell running your app has:

  ```bash
  export GOOGLE_GENAI_USE_VERTEXAI=true
  export VERTEXAI_PROJECT=avisia-training
  export VERTEXAI_LOCATION=europe-west1
  ```

  And that you did:

  ```bash
  gcloud auth application-default login
  gcloud auth application-default set-quota-project avisia-training
  ```

* **404 NOT_FOUND for model**
  Use `gemini-1.5-flash-002` (exists in `europe-west1`). The `gemini-2.0-flash-exp` preview may not be available.

* **MCP returns 400/404**

  * Use **dotted tool names**: `orders.get_status`, `returns.create_label`, etc.
  * Correct **argument names** (e.g., `"id"` for `orders.get_status` as in your gateway).
  * Invoke path: `/tools/invoke` (or your configured one).

* **ADK Web says “No agents found”**

  * Keep `adk_app.py` with `from agent import root_agent as AGENT` at project root.
  * Launch with `adk web apps` from project root and `PYTHONPATH=$PWD`.

* **Duplicated chat text or warnings**
  You already cleaned stream handling and added:

  ```bash
  export GOOGLE_GENAI_SUPPRESS_NON_TEXT_WARNING=1
  export GENAI_SUPPRESS_NON_TEXT_WARNING=1
  ```

---

## 12) Handy one-liners (copy/paste)

```bash
# Set Vertex env for THIS shell
export GOOGLE_GENAI_USE_VERTEXAI=true
export VERTEXAI_PROJECT=avisia-training
export VERTEXAI_LOCATION=europe-west1
export GOOGLE_CLOUD_PROJECT=$VERTEXAI_PROJECT
export GOOGLE_CLOUD_LOCATION=$VERTEXAI_LOCATION

# Verify Vertex works
python - <<'PY'
import os
from google import genai
c = genai.Client(vertexai=True,
                 project=os.environ["VERTEXAI_PROJECT"],
                 location=os.environ["VERTEXAI_LOCATION"])
print("OK?", bool(getattr(c.models.generate_content(
    model="gemini-1.5-flash-002", contents="ping"), "candidates", None)))
PY

# Start MCP locally
uvicorn mcp_server:app --host 0.0.0.0 --port 8080

# Point agents to MCP
export MCP_HTTP_URL="http://localhost:8080"

# Run ADK Web UI
export PYTHONPATH="$PWD"
adk web apps
```

---

## 13) License / Notes

* This repo uses experimental ADK features (you’ll see warnings in logs).
* Keep `requirements.txt` lean; add only what your tools need (Cloud Run/Agent Engine will pip install it).
* If you later add auth to the MCP gateway, set `MCP_HTTP_BEARER` and forward it in `agent.py` MCP calls.

---

**You’re set.** Next time: export the env block, start MCP (or point to Cloud Run), and use either the CLI or `adk web apps`.
