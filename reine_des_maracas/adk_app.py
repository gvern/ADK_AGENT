# adk_app.py
# Expose RUNNER to ensure ADK uses the pre-configured Artifact/Session services
from reine_des_maracas.services import runner as RUNNER
from reine_des_maracas.agent import root_agent as AGENT  # optional fallback
