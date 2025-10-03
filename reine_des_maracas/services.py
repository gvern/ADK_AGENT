# reine_des_maracas/services.py
import os
import logging
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService

logger = logging.getLogger(__name__)

# --- Définition des services et constantes ---
# Ces éléments ne dépendent pas de agent.py et peuvent être définis en premier.
APP_NAME = os.getenv("ADK_APP_NAME", "reine_des_maracas")
session_service_stateful = InMemorySessionService()
artifact_service_stateful = InMemoryArtifactService()

# --- Initialisation du Runner ---
# On déclare la variable runner, initialement à None.
runner = None
try:
    # LA MAGIE EST ICI : On importe root_agent JUSTE AVANT de l'utiliser.
    # À ce stade, le module agent.py a eu le temps d'être initialisé.
    from reine_des_maracas.agent import root_agent

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service_stateful,
        artifact_service=artifact_service_stateful,
    )
    logger.info("Runner ADK initialisé avec succès (via services.py).")

except ImportError as e:
    # Cette erreur se produira si agent.py a un problème de syntaxe.
    logger.error(f"Échec de l'importation de root_agent : {e}. L'importation circulaire est-elle résolue ?")
except Exception as e:
    logger.error(f"Échec de l'initialisation du Runner : {e}")