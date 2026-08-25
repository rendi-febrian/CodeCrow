"""Main entry point for the structural repository-index API."""

import logging
import os

try:
    from dotenv import load_dotenv
    load_dotenv(interpolate=False)
except Exception as dotenv_error:
    print(f"[ENV-BOOT] ERROR loading .env: {dotenv_error}", flush=True)

# New Relic must be initialized before application imports.
new_relic_config = os.environ.get("NEW_RELIC_CONFIG_FILE")
if new_relic_config and os.path.exists(new_relic_config):
    try:
        import newrelic.agent
        newrelic.agent.initialize(new_relic_config)
    except Exception as new_relic_error:
        print(
            f"[NR-BOOT] ERROR during initialization: {new_relic_error}",
            flush=True,
        )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_environment() -> None:
    """Report the only external store required by the structural index."""
    logger.info("Repository Index starting")
    logger.info("QDRANT_URL: %s", os.getenv("QDRANT_URL", "http://qdrant:6333"))
    logger.info(
        "QDRANT_COLLECTION_PREFIX: %s",
        os.getenv("QDRANT_COLLECTION_PREFIX", "codecrow"),
    )
    logger.info("Qdrant payload storage configured")


validate_environment()

import uvicorn
from rag_pipeline.api.api import app


if __name__ == "__main__":
    workers = int(os.environ.get("UVICORN_WORKERS", "1"))
    logger.info("Starting Uvicorn with %s worker process(es)", workers)
    uvicorn.run(
        "rag_pipeline.api.api:app",
        host="0.0.0.0",
        port=8001,
        workers=workers,
        interface="asgi3",
    )
