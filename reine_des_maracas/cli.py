#!/usr/bin/env python3
# ---- silence des warnings 'non-text parts' du SDK ----
import os, warnings, logging, sys, re

# 1) Variables d'env reconnues par certaines versions
os.environ.setdefault("GOOGLE_GENAI_SUPPRESS_NON_TEXT_WARNING", "1")
os.environ.setdefault("GENAI_SUPPRESS_NON_TEXT_WARNING", "1")

# 2) Filtre warnings.* (si c'est émis via warnings)
warnings.filterwarnings("ignore", message=r".*non-text parts.*", category=UserWarning)

# 3) Filtre logging.* (si c'est émis via logging)
for name in ("google.genai", "google.genai._response_utils"):
    logging.getLogger(name).setLevel(logging.ERROR)

# 4) (option dur) Filtre stdout si une lib print() en direct ce message
class _StdoutFilter:
    _rx = re.compile(r"non-text parts.*\['function_call'\]", re.IGNORECASE)
    def __init__(self, stream): self.stream = stream
    def write(self, s):
        if self._rx.search(s):  # drop la ligne
            return
        self.stream.write(s)
    def flush(self): self.stream.flush()

sys.stdout = _StdoutFilter(sys.stdout)
# ------------------------------------------------------
import os
import sys
import json
import argparse
from typing import Any, Dict

# Lightweight .env loader (no external deps)
def load_dotenv(path: str = ".env") -> None:
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass

load_dotenv()

try:
    import reine_des_maracas.agent as agent
except Exception as e:
    print(f"Failed to import agent.py: {e}", file=sys.stderr)
    sys.exit(1)


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def cmd_rag(args: argparse.Namespace) -> None:
    res = agent.rag_return_policy(args.question)
    _print_json(res)


def cmd_order_status(args: argparse.Namespace) -> None:
    res = agent.mcp_check_order_status(args.order_id)
    _print_json(res)


def cmd_return_label(args: argparse.Namespace) -> None:
    res = agent.mcp_create_return_label(args.order_id, args.reason)
    _print_json(res)


def cmd_search_products(args: argparse.Namespace) -> None:
    res = agent.mcp_search_products(args.query, size=args.size, color=args.color, limit=args.limit)
    _print_json(res)


def cmd_find_store(args: argparse.Namespace) -> None:
    stores = agent.gmaps_find_store(args.q, limit=args.limit)
    # Normalize dataclass to dict
    out = [s.__dict__ for s in stores]
    _print_json(out)


def cmd_store_hours(args: argparse.Namespace) -> None:
    res = agent.mcp_get_store_hours(args.store_id)
    _print_json(res)


def cmd_chat(args: argparse.Namespace) -> None:
    # Run one query through the Root agent if available
    if getattr(agent, "runner", None) is None:
        print("Runner not available. Ensure google-adk is installed and configured.", file=sys.stderr)
        sys.exit(2)
    import asyncio

    async def _run():
        user_id = os.getenv("USER_ID_STATEFUL", "gustave")
        session_id = os.getenv("SESSION_ID_STATEFUL", "cli-session-001")

        # Ensure the ADK session exists to avoid 'Session not found'
        svc = getattr(agent, "session_service_stateful", None)
        if svc is not None and getattr(agent, "APP_NAME", None):
            try:
                import inspect as _inspect
                sess = None
                if hasattr(svc, "get_session_sync"):
                    sess = svc.get_session_sync(app_name=agent.APP_NAME, user_id=user_id, session_id=session_id)
                else:
                    _get = getattr(svc, "get_session", None)
                    if _get is not None:
                        if _inspect.iscoroutinefunction(_get):
                            sess = await _get(app_name=agent.APP_NAME, user_id=user_id, session_id=session_id)
                        else:
                            sess = _get(app_name=agent.APP_NAME, user_id=user_id, session_id=session_id)
                if sess is None:
                    if hasattr(svc, "create_session_sync"):
                        svc.create_session_sync(app_name=agent.APP_NAME, user_id=user_id, session_id=session_id)
                    else:
                        _create = getattr(svc, "create_session", None)
                        if _create is not None:
                            if _inspect.iscoroutinefunction(_create):
                                await _create(app_name=agent.APP_NAME, user_id=user_id, session_id=session_id)
                            else:
                                _create(app_name=agent.APP_NAME, user_id=user_id, session_id=session_id)
            except Exception:
                # Best-effort; runner may still handle creation depending on version
                pass

        resp = await agent.call_agent_async(args.ask, agent.runner, user_id, session_id)


    asyncio.run(_run())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reine des Maracas – MCP/RAG/Maps CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("rag", help="Ask the returns/refunds policy RAG")
    sp.add_argument("question", help="Question to ask the policy RAG")
    sp.set_defaults(func=cmd_rag)

    sp = sub.add_parser("order-status", help="Check order status via MCP")
    sp.add_argument("order_id", help="Order ID")
    sp.set_defaults(func=cmd_order_status)

    sp = sub.add_parser("return-label", help="Create a return label via MCP")
    sp.add_argument("order_id", help="Order ID")
    sp.add_argument("reason", help="Reason for return")
    sp.set_defaults(func=cmd_return_label)

    sp = sub.add_parser("search-products", help="Search products via MCP")
    sp.add_argument("query", help="Search query")
    sp.add_argument("--size", dest="size")
    sp.add_argument("--color", dest="color")
    sp.add_argument("--limit", dest="limit", type=int, default=5)
    sp.set_defaults(func=cmd_search_products)

    sp = sub.add_parser("find-store", help="Search stores near a city/address using Google Places")
    sp.add_argument("-q", "--q", required=True, help="City or address")
    sp.add_argument("--limit", type=int, default=5)
    sp.set_defaults(func=cmd_find_store)

    sp = sub.add_parser("store-hours", help="Get store hours via MCP")
    sp.add_argument("store_id", help="Store ID (e.g., Place ID)")
    sp.set_defaults(func=cmd_store_hours)

    sp = sub.add_parser("chat", help="Ask the Root agent (requires google-adk)")
    sp.add_argument("--ask", required=True, help="User query to the agent")
    sp.set_defaults(func=cmd_chat)

    return p


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
