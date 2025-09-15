# deploy.py
import vertexai
from vertexai import agent_engines
from reine_des_maracas.agent import root_agent  # <-- importe le root que tu as déjà créé

PROJECT_ID     = "ton-projet"
LOCATION       = "europe-west1"          # choisis une région supportée
STAGING_BUCKET = "gs://ton-bucket-staging"

vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

# 1) wrap ADK -> AdkApp (trace activé utile en prod)
app = agent_engines.AdkApp(agent=root_agent, enable_tracing=True)

# 2) variables d'env pour le runtime managé (ex: URLs, clés en Secret Manager via refs)
env_vars = {
  "MCP_HTTP_URL": "https://mcp-gateway.prod.company.com",   # ou laisse vide si pas de MCP
  "GOOGLE_MAPS_API_KEY": "projects/123456789/secrets/gmaps_key/versions/latest"  # si tu utilises Secret Manager
}

# 3) options de ressource (auto-scaling)
build_options = {"timeout_sec": 1800}  # ex. 30 min pour installer les deps
resource_limits = {"cpu": "2", "memory": "4Gi"}  # ajuste selon ton agent

remote_app = agent_engines.create(
    agent_engine=app,
    requirements=requirements,
    config={
      "env_vars": env_vars,
      "display_name": "Reine des Maracas (root)",
      "description": "Root agent + sub-agents SAV/Sales (MCP + GMaps)",
      "min_instances": 0,
      "max_instances": 3,
      "resource_limits": resource_limits,
      "build_options": build_options,
      # "service_account": "sa-agent-runtime@ton-projet.iam.gserviceaccount.com",  # si besoin dédié
    },
)

print("✅ Deploy OK")
print("Resource Name:", remote_app.resource_name)  # projects/NUM/locations/REGION/reasoningEngines/ID
