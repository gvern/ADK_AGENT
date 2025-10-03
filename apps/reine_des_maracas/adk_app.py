# Prefer exposing RUNNER so ADK uses our pre-configured services (Artifact/Session)
from reine_des_maracas.services import runner as RUNNER
from reine_des_maracas.agent import root_agent as AGENT  # optional fallback
