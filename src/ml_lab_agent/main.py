import logging
import sys

from fastapi import FastAPI

from ml_lab_agent.api.routes.chat import chat_router
from ml_lab_agent.api.routes.experiments import experiment_router
from ml_lab_agent.api.routes.health import health_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(experiment_router)
    app.include_router(chat_router)

    return app


app = create_app()

logger.info("MLLabAgent app initialized.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ml_lab_agent.main:app", host="127.0.0.1", port=8000, reload=True, log_level="info", access_log=True)
