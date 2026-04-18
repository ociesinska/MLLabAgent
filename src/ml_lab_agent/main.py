from fastapi import FastAPI

from ml_lab_agent.api.routes.experiments import experiment_router
from ml_lab_agent.api.routes.health import health_router


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(experiment_router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ml_lab_agent.main:app", host="127.0.0.1", port=8000, reload=True)
