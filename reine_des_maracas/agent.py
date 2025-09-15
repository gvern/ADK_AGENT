# @title 1) Define the before_model_callback Guardrail (keyword block)

from typing import Optional, Dict, Any, List, cast
from google.genai import types
import os
import json
from urllib.parse import urlencode, quote_plus
import math
import logging
try:
    import requests  # type: ignore
except Exception:  # requests might not be installed in some envs
    requests = None  # we'll guard at call sites

# Try Google ADK/GenAI imports, fallback to lightweight stubs for local importability
try:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types  # for response content
except Exception:
    class CallbackContext:  # type: ignore
        def __init__(self, agent_name: str = "agent", state: Optional[Dict[str, Any]] = None):
            self.agent_name = agent_name
            self.state = state or {}

    class _TypesPart:
        def __init__(self, text: str = ""):
            self.text = text

    class _TypesContent:
        def __init__(self, role: str = "model", parts: Optional[List[Any]] = None):
            self.role = role
            self.parts = parts or []

    class types:  # type: ignore
        Content = _TypesContent
        Part = _TypesPart

    class LlmRequest:  # type: ignore
        def __init__(self, contents: Optional[List[Any]] = None):
            self.contents = contents or []

    class LlmResponse:  # type: ignore
        def __init__(self, content: Any = None):
            self.content = content
import os
from google import genai
import os
from google import genai

# --- Bootstrap Vertex AI (respecte tes exports VERTEXAI_PROJECT/LOCATION) ---
PROJECT  = os.environ.get("VERTEXAI_PROJECT", "avisia-training")
LOCATION = os.environ.get("VERTEXAI_LOCATION", "europe-west1")
if not PROJECT or not LOCATION:
    raise RuntimeError("VERTEXAI_PROJECT et VERTEXAI_LOCATION doivent être définies")

GENAI_VERTEX_CLIENT = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

os.environ.setdefault("VERTEXAI_PROJECT", "avisia-training")
os.environ.setdefault("VERTEXAI_LOCATION", "europe-west1")
_genai = genai.Client(vertexai=True,
                      project=os.environ["VERTEXAI_PROJECT"],
                      location=os.environ["VERTEXAI_LOCATION"])


# fallback si tu gardes tes anciens noms .env
os.environ.setdefault("VERTEXAI_PROJECT", os.getenv("VERTEX_PROJECT", "avisia-training"))
os.environ.setdefault("VERTEXAI_LOCATION", os.getenv("VERTEX_LOCATION", "europe-west1"))

# construit explicitement un client Vertex
_vertex = genai.Client(
    vertexai=True,
    project=os.environ["VERTEXAI_PROJECT"],
    location=os.environ["VERTEXAI_LOCATION"],
)

def block_keyword_guardrail(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Guardrail: bloque la requête si elle contient le mot-clé 'BLOCK' (EN) ou 'BLOQUE' (FR).
    Sinon, laisse passer vers le LLM.
    """
    agent_name = callback_context.agent_name
    print(f"--- Callback: block_keyword_guardrail for agent: {agent_name} ---")

    last_user_message_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts and getattr(content.parts[0], "text", None):
                last_user_message_text = content.parts[0].text
                break

    print(f"--- Inspecting last user message: '{last_user_message_text[:120]}...' ---")

    blocked_terms = {"BLOCK", "BLOQUE"}
    if any(term in last_user_message_text.upper() for term in blocked_terms):
        callback_context.state["guardrail_block_keyword_triggered"] = True
        print(f"--- Found a blocked keyword. Blocking LLM call. ---")
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="Demande bloquée par la politique de sécurité (mot-clé interdit).")],
            )
        )
    return None

print("✅ block_keyword_guardrail function defined.")

# @title 2) Define Tools (stubs) for SAV & Sales — swap with your MCP/RAG/GMaps

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# ── SAV TOOLS ────────────────────────────────────────────────────────────────
# 1) RAG: politique de retour / remboursement (remplacer par ton store vectoriel)
def rag_return_policy(question: str) -> Dict[str, Any]:
    """
    RAG pour la politique de retours/remboursements « Reine des Maracas ».

    Ordre des backends (auto-détecté):
    1) MCP → agentspace.search (Vertex AI Search/Agentspace) avec BigQuery (avisia-training.reine_des_maracas, europe-west1)
       - Configurez MCP_HTTP_URL pour pointer vers votre passerelle MCP.
       - Optionnel: AGENTSPACE_URL pour préciser l'Agentspace (sinon valeur par défaut ci-dessous).
    2) CHROMA (si CHROMA_PATH et chromadb installés) → collection « reine_des_maracas_policies »
    3) Fallback local (matching Jaccard) pour garder le script fonctionnel.

    Retourne: {"answer": str, "sources": List[str]}
    """

    q = (question or "").strip()
    if not q:
        return {"answer": "Veuillez préciser votre question sur les retours/remboursements.", "sources": []}

    # 1) MCP Agentspace (Vertex AI Search)
    try:
        agentspace_url_env = os.getenv(
            "AGENTSPACE_URL",
            # Valeur fournie par l'utilisateur (UI). L'outil MCP saura l'utiliser si pertinent.
            "https://vertexaisearch.cloud.google.com/eu/home/cid/6a59ad39-5ee7-41c7-ae22-021eb8cc1998?hl=en_GB",
        )
        project = os.getenv("VERTEX_PROJECT", "avisia-training")
        dataset = os.getenv("VERTEX_BQ_DATASET", "reine_des_maracas")
        location = os.getenv("VERTEX_LOCATION", "europe-west1")
        bq_ctx = _bigquery_ctx()
        # Si une passerelle MCP est configurée, on tente agentspace.search
        if os.getenv("MCP_HTTP_URL"):
            res = _call_mcp_tool(
                "agentspace.search",
                {
                    "query": q,
                    "agentspace_url": agentspace_url_env,
                    "project": project,
                    "location": location,
                    "bigquery": bq_ctx,
                    "top_k": 3,
                    "scope": ["returns", "refunds", "policy"],
                },
            )
            if isinstance(res, dict) and "answer" in res:
                ans = str(res.get("answer") or "")
                sources = res.get("sources") or res.get("documents") or []
                if isinstance(sources, list):
                    srcs = [str(s) for s in sources]
                else:
                    srcs = [str(sources)]
                if ans:
                    return {"answer": ans, "sources": srcs}
    except Exception as e:
        logging.warning(f"agentspace.search via MCP a échoué; fallback: {e}")

    # 2) CHROMA backend (optionnel)
    try:
        chroma_path = os.getenv("CHROMA_PATH")
        if chroma_path:
            import chromadb  # type: ignore
            client = chromadb.PersistentClient(path=chroma_path)
            coll_name = os.getenv("CHROMA_COLLECTION", "reine_des_maracas_policies")
            coll = client.get_or_create_collection(name=coll_name)
            # On suppose que la collection a été créée avec une embedding_function compatible
            res = coll.query(query_texts=[q], n_results=3)
            docs = (res.get("documents") or [[]])[0]
            metadatas = (res.get("metadatas") or [[]])[0]
            if docs:
                answer = docs[0]
                sources: List[str] = []
                for md in metadatas:
                    src = (md or {}).get("source") if isinstance(md, dict) else None
                    if src:
                        sources.append(str(src))
                if not sources:
                    # fallback metadata-less: use collection/id references
                    ids = (res.get("ids") or [[]])[0]
                    sources = [f"chroma://{coll_name}/{i}" for i in ids]
                return {"answer": str(answer), "sources": sources[:3]}
    except Exception as e:
        logging.warning(f"CHROMA RAG indisponible; fallback: {e}")

    # 3) Fallback local: mini-KB + Jaccard
    KB_DOCS: List[Dict[str, str]] = [
        {
            "id": "returns_policy_v1",
            "source": "kb://returns_policy_v1",
            "text": (
                "Politique de retours: Vous disposez de 30 jours après réception pour retourner un article. "
                "Le produit doit être neuf, non porté, avec étiquettes et dans son emballage d'origine. "
                "Après validation en entrepôt, le remboursement est effectué sous 5 à 7 jours ouvrés. "
                "Les frais de retour sont offerts pour les retours depuis la France métropolitaine."
            ),
        },
        {
            "id": "refunds_faq_v2",
            "source": "kb://faq_refunds_v2",
            "text": (
                "Remboursements: Les remboursements sont crédités sur le moyen de paiement initial. "
                "Délais typiques: 5 à 7 jours ouvrés après validation. Les échanges sont possibles selon stock."
            ),
        },
    ]

    def _tokenize(s: str) -> set:
        return {t.lower() for t in ''.join(ch if ch.isalnum() else ' ' for ch in s).split() if t}

    q_tokens = _tokenize(q)
    best = None
    best_score = -1.0
    for d in KB_DOCS:
        d_tokens = _tokenize(d["text"])
        inter = len(q_tokens & d_tokens)
        union = len(q_tokens | d_tokens) or 1
        score = inter / union
        if score > best_score:
            best = d
            best_score = score

    if best:
        return {"answer": best["text"], "sources": [best["source"]]}
    combined = " \n\n".join(d["text"] for d in KB_DOCS)
    return {"answer": combined, "sources": [d["source"] for d in KB_DOCS]}

# 2) MCP: statut de commande (expose ton endpoint via MCP server)
def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Generic MCP HTTP invocation.

    Configure via env:
      - MCP_HTTP_URL: Base URL of your MCP HTTP gateway (e.g., https://mcp.mycompany.com)
      - MCP_HTTP_INVOKE_PATH: Path to invoke tools (default: /tools/invoke)

    Payload expected by gateway: {"tool": str, "arguments": dict}
    Response expected: JSON dict with either direct fields or under 'result'.
    """
    base = os.getenv("MCP_HTTP_URL")
    if not base:
        raise RuntimeError("MCP_HTTP_URL not set; can't invoke MCP tools. Configure an HTTP gateway or add a stdio client.")
    if requests is None:
        raise RuntimeError("'requests' is not available; install it to call MCP HTTP gateway.")
    path = os.getenv("MCP_HTTP_INVOKE_PATH", "/tools/invoke")
    url = base.rstrip("/") + path
    headers = {}
    bearer = os.getenv("MCP_HTTP_BEARER")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    try:
        r = requests.post(url, json={"tool": tool_name, "arguments": arguments}, headers=headers or None, timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "result" in data:
            return data["result"]
        return data
    except Exception as e:
        raise RuntimeError(f"MCP call failed for {tool_name}: {e}")

def _bigquery_ctx() -> Dict[str, Any]:
    project = os.getenv("VERTEX_PROJECT", "avisia-training")
    dataset = os.getenv("VERTEX_BQ_DATASET", "reine_des_maracas")
    location = os.getenv("VERTEX_LOCATION", "europe-west1")
    # If explicit table list provided, use it; else default to known tables
    tables_env = os.getenv("VERTEX_BQ_TABLES")
    if tables_env:
        tables = [t.strip() for t in tables_env.split(",") if t.strip()]
    else:
        tables = [
            "complement_individu",
            "individu",
            "magasin",
            "referentiel",
            "ticket_caisse",
            "typo_produit",
        ]
    return {
        "project": project,
        "dataset": dataset,
        "location": location,
        "tables": tables,
    }

def mcp_check_order_status(order_id: str) -> Dict[str, Any]:
    """
    MCP tool: orders.get_status
    """
    try:
        args = {"order_id": order_id, "bigquery": _bigquery_ctx()}
        res = _call_mcp_tool("orders.get_status", args)
        if isinstance(res, dict):
            return cast(Dict[str, Any], res)
        # If gateway returns a list or wrapped shape
        return {"order_id": order_id, **(res or {})}
    except Exception as e:
        logging.warning(f"orders.get_status MCP failed, using fallback: {e}")
        return {"order_id": order_id, "status": "Delivered", "delivered_at": "2025-09-04"}

# 3) MCP: créer une étiquette de retour
def mcp_create_return_label(order_id: str, reason: str) -> Dict[str, Any]:
    """
    MCP tool: returns.create_label
    """
    try:
        args = {"order_id": order_id, "reason": reason, "bigquery": _bigquery_ctx()}
        res = _call_mcp_tool("returns.create_label", args)
        if isinstance(res, dict):
            return cast(Dict[str, Any], res)
        return {"order_id": order_id, "reason": reason, **(res or {})}
    except Exception as e:
        logging.warning(f"returns.create_label MCP failed, using fallback: {e}")
        return {"order_id": order_id, "label_url": f"https://returns.example/label/{order_id}", "reason": reason}


# ── SALES TOOLS ─────────────────────────────────────────────────────────────
# 1) MCP: recherche produits (catégorie, filtres)
def mcp_search_products(query: str, size: Optional[str] = None, color: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    MCP tool: catalog.search
    """
    args: Dict[str, Any] = {"query": query, "limit": limit, "bigquery": _bigquery_ctx()}
    if size:
        args["size"] = size
    if color:
        args["color"] = color
    try:
        res = _call_mcp_tool("catalog.search", args)
        if isinstance(res, list):
            return cast(List[Dict[str, Any]], res)[:limit]
        if isinstance(res, dict) and "items" in res and isinstance(res["items"], list):
            return cast(List[Dict[str, Any]], res["items"])[:limit]
        # Coerce single dict
        if isinstance(res, dict):
            return [cast(Dict[str, Any], res)]
    except Exception as e:
        logging.warning(f"catalog.search MCP failed, using fallback: {e}")
    return [
        {"sku": "BKN-TRI-001", "title": "Bikini triangle noir", "size": size or "S", "color": color or "noir", "price_eur": 39.9, "availability": "in_stock"},
        {"sku": "SWM-ONE-PIECE-RED", "title": "Maillot 1 pièce rouge", "size": size or "M", "color": color or "rouge", "price_eur": 59.0, "availability": "low_stock"},
    ][:limit]

# 2) Google Maps: recherche magasins proches (mock)
@dataclass
class Store:
    name: str
    address: str
    city: str
    place_id: str
    distance_km: float

# Remplace la signature et les retours de gmaps_find_store par ceci
from typing import Optional, Dict, Any, List
import os, math, logging

try:
    import requests  # si tu l’as déjà importé ailleurs, laisse comme c’est
except Exception:
    requests = None

def gmaps_find_store(city_or_address: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Google Places API search for nearby stores matching 'Reine des Maracas'.

    Args:
        city_or_address: City name or full address used as origin for distance.
        limit: Max results to return (default: 5).

    Returns:
        List[Dict[str, Any]] with JSON-serializable items:
        [
          {
            "name": str,
            "address": str,
            "city": str,
            "place_id": str,
            "distance_km": float
          },
          ...
        ]

    Note:
        Only JSON-friendly types are used to comply with ADK auto function calling.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key or requests is None:
        logging.warning("GOOGLE_MAPS_API_KEY not set or requests unavailable; using mock stores.")
        return [
            {
                "name": "Reine des Maracas — Opéra",
                "address": "12 Rue de la Paix, 75002 Paris",
                "city": "Paris",
                "place_id": "gmaps:opera-123",
                "distance_km": 1.2,
            },
            {
                "name": "Reine des Maracas — Marais",
                "address": "5 Rue des Rosiers, 75004 Paris",
                "city": "Paris",
                "place_id": "gmaps:marais-456",
                "distance_km": 2.4,
            },
        ][:limit]

    def _geocode(q: str) -> Optional[Dict[str, float]]:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": q, "key": api_key}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                return {"lat": float(loc["lat"]), "lng": float(loc["lng"])}
        except Exception:
            return None
        return None

    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(R * c, 2)

    origin = _geocode(city_or_address)
    text_query = f"Reine des Maracas near {city_or_address}"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": text_query, "key": api_key}
    results: List[Dict[str, Any]] = []

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        for item in (data.get("results") or [])[:limit]:
            name = item.get("name") or "Reine des Maracas"
            address = item.get("formatted_address") or item.get("vicinity") or ""
            place_id = item.get("place_id") or ""
            lat = (item.get("geometry") or {}).get("location", {}).get("lat")
            lng = (item.get("geometry") or {}).get("location", {}).get("lng")
            dist = 0.0
            if origin and isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                dist = _haversine_km(origin["lat"], origin["lng"], float(lat), float(lng))

            # Récupération éventuelle de la ville via Place Details
            city = ""
            if place_id:
                try:
                    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                    details_params = {
                        "place_id": place_id,
                        "key": api_key,
                        "fields": "address_components,formatted_address",
                    }
                    dr = requests.get(details_url, params=details_params, timeout=10)
                    dr.raise_for_status()
                    ddata = dr.json()
                    result = ddata.get("result") or {}
                    address = result.get("formatted_address") or address
                    comps = result.get("address_components") or []
                    for comp in comps:
                        if "locality" in comp.get("types", []):
                            city = comp.get("long_name") or city
                except Exception:
                    pass

            if not city and isinstance(address, str) and "," in address:
                city = address.split(",")[-1].strip()

            results.append(
                {
                    "name": name,
                    "address": address,
                    "city": city,
                    "place_id": place_id,
                    "distance_km": dist,
                }
            )

        if results:
            return results

    except Exception as e:
        logging.warning(f"Google Places API failed, using mock: {e}")

    # Fallback mock si l’API échoue
    return [
        {
            "name": "Reine des Maracas — Opéra",
            "address": "12 Rue de la Paix, 75002 Paris",
            "city": "Paris",
            "place_id": "gmaps:opera-123",
            "distance_km": 1.2,
        },
        {
            "name": "Reine des Maracas — Marais",
            "address": "5 Rue des Rosiers, 75004 Paris",
            "city": "Paris",
            "place_id": "gmaps:marais-456",
            "distance_km": 2.4,
        },
    ][:limit]

# 3) MCP: horaires magasin
def mcp_get_store_hours(store_id: str) -> Dict[str, Any]:
    """
    MCP tool: stores.hours
    """
    try:
        args = {"store_id": store_id, "bigquery": _bigquery_ctx()}
        res = _call_mcp_tool("stores.hours", args)
        if isinstance(res, dict):
            return cast(Dict[str, Any], res)
        return {"store_id": store_id, **(res or {})}
    except Exception as e:
        logging.warning(f"stores.hours MCP failed, using fallback: {e}")
        hours = {
            "mon-fri": "10:00–19:30",
            "sat": "10:00–20:00",
            "sun": "11:00–18:00",
        }
        return {"store_id": store_id, "hours": hours}

# @title 3) Define Agents (SAV, Sales) + Root with guardrail

# Assumes these are available from your environment; define defaults if not set.
MODEL_GEMINI_2_0_FLASH = "gemini-1.5-flash-002"

try:
    from google.adk import Agent, Runner  # type: ignore
except Exception:
    Agent = None  # type: ignore
    Runner = None  # type: ignore

APP_NAME = "reine-des-maracas"
print(f"✅ Using model: {MODEL_GEMINI_2_0_FLASH} | APP_NAME={APP_NAME}")

# ── SAV Agent ───────────────────────────────────────────────────────────────
if Agent is not None:
    sav_agent = Agent(
        model=MODEL_GEMINI_2_0_FLASH,
        name="sav_agent",
        description="Service Après-Vente Reine des Maracas: retours, remboursements, suivi commande.",
        instruction=(
            "Tu es l'agent SAV pour la boutique Reine des Maracas (maillots, sous-vêtements). "
            "Réponds précisément aux questions sur retours/remboursements/suivi commande. "
            "Utilise en priorité 'rag_return_policy' pour la politique officielle et cite les sources. "
            "Pour le suivi/retour, utilise 'mcp_check_order_status' et 'mcp_create_return_label'. "
            "Si la demande sort du périmètre SAV, explique poliment et renvoie au Root Agent."
        ),
        tools=[rag_return_policy, mcp_check_order_status, mcp_create_return_label],
    )

# ── Sales Agent ────────────────────────────────────────────────────────────
if Agent is not None:
    sales_agent = Agent(
        model=MODEL_GEMINI_2_0_FLASH,
        name="sales_agent",
        description="Vente & magasins Reine des Maracas: recherche produit, disponibilité, adresses magasins.",
        instruction=(
            "Tu es l'agent Sales de Reine des Maracas. "
            "Aide à trouver des produits (taille, couleur, prix) via 'mcp_search_products'. "
            "Pour trouver des magasins et adresses, utilise 'gmaps_find_store' puis, si utile, 'mcp_get_store_hours'. "
            "Structure les résultats (top 3), fais des suggestions (upsell/cross-sell) avec tact, "
            "et propose un plan d’action clair (acheter en ligne, réserver en boutique, etc.)."
        ),
        tools=[mcp_search_products, gmaps_find_store, mcp_get_store_hours],
    )

    print(f"✅ Sub-Agents ready: {sav_agent.name}, {sales_agent.name}")

# ── Root Agent ─────────────────────────────────────────────────────────────
if Agent is not None:
    root_agent = Agent(
        name="root_agent_reine_des_maracas",
        model=MODEL_GEMINI_2_0_FLASH,
        description=(
            "Agent racine qui route les demandes vers SAV (retours/remboursements/suivi) "
            "ou Sales (produits/magasins)."
        ),
        instruction=(
            "Tu es le Root Agent de Reine des Maracas. "
            "1) Détecte l'intention: 'SAV' (retours, remboursement, suivi commande, étiquette de retour) "
            "ou 'Sales' (recherche produit, tailles, couleurs, prix, magasins/adresses/horaires). "
            "2) Délègue intégralement à l'agent adéquat (sav_agent, sales_agent). "
            "3) Ne duplique pas l'information: si un sous-agent répond, synthétise brièvement et conclus par les prochaines étapes."
        ),
        sub_agents=[sav_agent, sales_agent],
        before_model_callback=block_keyword_guardrail,
        output_key="last_root_answer",
    )

    print(f"✅ Root Agent created: {root_agent.name}")

# @title 4) Runner (stateful) + quick test convo
# Si tu as déjà un session_service_stateful (de tes steps précédents), on le réutilise.
# Sinon, on fait un runner sans service (stateless) pour le test local.

# Resolve session service: reuse if provided globally, else create ADK InMemorySessionService, else go stateless
session_service_stateful = globals().get("session_service_stateful", None)
if session_service_stateful is None:
    try:
        from google.adk.sessions import InMemorySessionService  # type: ignore
        session_service_stateful = InMemorySessionService()
        HAS_STATEFUL = True
    except Exception:
        session_service_stateful = None  # type: ignore
        HAS_STATEFUL = False
else:
    HAS_STATEFUL = True

# IDs de session/utilisateur (configurables via env variables)
USER_ID_STATEFUL = os.getenv("USER_ID_STATEFUL", "gustave")
SESSION_ID_STATEFUL = os.getenv("SESSION_ID_STATEFUL", "demo-sessions-001")

if Runner is not None and Agent is not None:
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service_stateful if HAS_STATEFUL else None
    )
    print(f"✅ Runner ready (stateful={HAS_STATEFUL}).")
else:
    runner = None  # type: ignore
    print("⚠️ Google ADK non disponible: agents/runner non initialisés. Installez google-adk pour les exécuter.")

# Helper async caller (identique à ton pattern)
import io
import re
import contextlib
from google.genai import types

_code_inline_rx = re.compile(r"`[^`]*`")

def _canon(s: str) -> str:
    s = _code_inline_rx.sub("", s)
    s = " ".join(s.split()).strip()
    return s.lower()

def _collect_best_text(content) -> str:
    if not getattr(content, "parts", None):
        return ""
    texts, seen = [], set()
    for p in content.parts:
        t = getattr(p, "text", None)
        if not t:
            continue  # ignore function_call & co
        norm = _canon(t)
        if norm and norm not in seen:
            seen.add(norm)
            texts.append((" ".join(t.split()).strip(), len(norm)))
    if not texts:
        return ""
    texts.sort(key=lambda x: x[1], reverse=True)  # garde le bloc le plus long
    return texts[0][0]

async def call_agent_async(query: str, runner_obj, user_id: str, session_id: str):
    print(f"\n[USER] {query}")
    if runner_obj is None:
        print("[AGENT] (runner indisponible)")
        return None

    content = types.Content(role="user", parts=[types.Part(text=query)])
    resp = None

    # --- coupe TOUT ce que le SDK/ADK pourrait imprimer (warnings, traces, drafts) ---
    _sink_out, _sink_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(_sink_out), contextlib.redirect_stderr(_sink_err):
        try:
            # Mode flux d'événements (sans afficher le flux)
            events = runner_obj.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
            )
            final_event = None
            async for ev in events:
                if callable(getattr(ev, "is_final_response", None)) and ev.is_final_response():
                    final_event = ev
                    break
            resp = final_event
        except TypeError:
            # Variantes de signature selon versions
            try:
                resp = await runner_obj.run_async(
                    user_id=user_id, session_id=session_id, new_message=content
                )
            except TypeError:
                try:
                    resp = await runner_obj.run_async(
                        user_id=user_id, session_id=session_id, input=query
                    )
                except TypeError:
                    resp = await runner_obj.run_async(query)
    # -------------------------------------------------------------------------------

    final_text = ""
    if resp and getattr(resp, "content", None):
        # print(f"[DEBUG] parts: text={sum(1 for p in resp.content.parts if getattr(p,'text',None))} non-text={len(resp.content.parts)-sum(1 for p in resp.content.parts if getattr(p,'text',None))}")
        final_text = _collect_best_text(resp.content)

    print(f"[AGENT] {final_text or '(pas de texte – réponse outil?)'}")
    return resp



# Mini script de test: 1 SAV, 1 blocage, 1 Sales
async def run_demo():
    interaction = lambda q: call_agent_async(q, runner, USER_ID_STATEFUL, SESSION_ID_STATEFUL)

    print("\n--- DEMO: Root → SAV ---")
    await interaction("Je veux retourner un maillot, comment obtenir une étiquette de retour ?")

    print("\n--- DEMO: Guardrail BLOCK ---")
    await interaction("BLOCK cette requête pour un remboursement immédiat")

    print("\n--- DEMO: Root → Sales ---")
    await interaction("Je cherche un bikini noir en taille M près de Paris. Une adresse de magasin ?")

    # End of demo

    if HAS_STATEFUL and runner is not None and session_service_stateful is not None:
        # Optionnel : inspection d'état (compatible avec ADK Session model)
        sess = await session_service_stateful.get_session(app_name=APP_NAME, user_id=USER_ID_STATEFUL, session_id=SESSION_ID_STATEFUL)  # type: ignore[attr-defined]
        if sess is not None:
            print("\n--- Session State Snapshot ---")
            st = getattr(sess, "state", None)
            state_map: Dict[str, Any] = {}
            if isinstance(st, dict):
                state_map = st
            elif st is not None:
                # ADK typically exposes a pydantic SessionState model with `.data` as dict
                state_map = getattr(st, "data", {}) if isinstance(getattr(st, "data", {}), dict) else {}
            print("guardrail_block_keyword_triggered:", state_map.get("guardrail_block_keyword_triggered"))
            print("last_root_answer:", state_map.get("last_root_answer"))
