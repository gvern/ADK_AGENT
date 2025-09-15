# mcp_server.py — Gateway HTTP tolérant (noms . et _, alias d'arguments)
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, Callable, List, Optional
import inspect


app = FastAPI(title="Reine des Maracas MCP Gateway", version="0.1.0")

# --------- Implémentations outils ---------
def orders_get_status(order_id: str) -> Dict[str, Any]:
    return {"order_id": order_id, "status": "Delivered", "delivered_at": "2025-09-04"}

def returns_create_label(order_id: str, reason: str) -> Dict[str, Any]:
    return {"order_id": order_id, "label_url": f"https://returns.example/label/{order_id}", "reason": reason}

def catalog_search(query: str, size: Optional[str] = None, color: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    data = [
        {"sku": "BKN-TRI-001", "title": "Bikini triangle noir", "size": size or "M", "color": color or "noir", "price_eur": 39.9, "availability": "in_stock"},
        {"sku": "SWM-ONE-PIECE-RED", "title": "Maillot 1 pièce rouge", "size": size or "M", "color": color or "rouge", "price_eur": 59.0, "availability": "low_stock"},
    ]
    return data[:limit]

def stores_hours(store_id: str) -> Dict[str, Any]:
    return {"store_id": store_id, "hours": {"mon-fri": "10:00–19:30", "sat": "10:00–20:00", "sun": "11:00–18:00"}}
def agentspace_search(query: str, top_k: int = 4):
    # Mock simple de RAG : renvoie des “sources” kb avec scores
    base = [
        {"id": "kb://returns_policy_v1", "title": "Politique de retours", "score": 0.92},
        {"id": "kb://shipping_policy_v1", "title": "Livraison & délais", "score": 0.81},
        {"id": "kb://size_guide_v2",     "title": "Guide des tailles",    "score": 0.78},
        {"id": "kb://care_swimwear_v1",  "title": "Entretien maillots",   "score": 0.71},
    ]
    return base[: int(top_k)]


# --------- Registre canonique (underscores) ---------
TOOLS_CANON: Dict[str, Callable[..., Any]] = {
    "orders_get_status": orders_get_status,
    "returns_create_label": returns_create_label,
    "catalog_search": catalog_search,
    "stores_hours": stores_hours,
    "agentspace_search": agentspace_search,
}

# --------- Alias noms (dotted -> underscore) ---------
ALIAS_NAME: Dict[str, str] = {
    "orders.get_status": "orders_get_status",
    "returns.create_label": "returns_create_label",
    "catalog.search": "catalog_search",
    "stores.hours": "stores_hours",
    "agentspace.search": "agentspace_search",
}
ALIAS_INV = {v: k for k, v in ALIAS_NAME.items()}

# --------- Alias d'arguments par outil ---------

ARG_ALIASES: Dict[str, Dict[str, str]] = {
    "orders_get_status": {
        "order_id": "order_id",
        "id": "order_id",
        "orderId": "order_id",
        "order-id": "order_id",
    },
    "returns_create_label": {
        "order_id": "order_id",
        "id": "order_id",
        "orderId": "order_id",
        "order-id": "order_id",
        "reason": "reason",
        "motif": "reason",
        "return_reason": "reason",
        "labelReason": "reason",
    },
    "catalog_search": {
        "query": "query",
        "q": "query",
        "size": "size",
        "color": "color",
        "limit": "limit",
        "n": "limit",
    },
    "stores_hours": {
        "store_id": "store_id",
        "place_id": "store_id",
        "placeId": "store_id",
        "storeId": "store_id",
        "store-id": "store_id",
    },
    "agentspace_search": {                      # <--- nouveau
        "q": "query",
        "topK": "top_k",
        "topk": "top_k",
    },
}

IGNORED_EXTRAS = {"bigquery"}

def normalize_args(tool_canon: str, args: Dict[str, Any], fn) -> Dict[str, Any]:
    args = args or {}
    # 1) mappe alias -> canon
    mapping = ARG_ALIASES.get(tool_canon, {})
    mapped = {}
    for k, v in args.items():
        if k in IGNORED_EXTRAS:
            # on ignore les extras (ex: bigquery) envoyés par l'ADK
            continue
        mapped[mapping.get(k, k)] = v

    # 2) conversions simples
    if tool_canon == "catalog_search" and "limit" in mapped:
        try:
            mapped["limit"] = int(mapped["limit"])
        except Exception:
            raise HTTPException(status_code=400, detail="limit must be an integer")

    # 3) filtre final par signature de la fonction
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in mapped.items() if k in allowed}
    return filtered

class InvokeBody(BaseModel):
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = {}

@app.get("/health")
def health():
    return {"status": "ok", "server": "reine-des-maracas"}

@app.get("/tools")
def list_tools():
    canon = sorted(list(TOOLS_CANON.keys()))
    dotted = sorted([ALIAS_INV.get(n, n) for n in canon])
    return {"tools": canon, "dotted_tools": dotted}

def _resolve_tool(name: str) -> (str, Callable[..., Any]):
    canon_name = ALIAS_NAME.get(name, name)
    fn = TOOLS_CANON.get(canon_name)
    if not fn:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {name}")
    return canon_name, fn

# /tools/invoke — accepte tool_name OU tool (compat ADK)
@app.post("/tools/invoke")
async def tools_invoke(req: Request):
    data = await req.json()
    print("[MCP] invoke payload:", data)
    name = data.get("tool_name") or data.get("tool")
    args = data.get("arguments") or {}
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'tool' or 'tool_name'")
    canon_name, fn = _resolve_tool(name)
    norm_args = normalize_args(canon_name, args, fn)  # <-- ici
    try:
        return {"result": fn(**norm_args)}
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Bad arguments for {name}: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool call failed: {e}")

