import asyncio
import signal
import sys

import fastapi
import sqlmodel
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from oasst_inference_server import database, deps, models, plugins
from oasst_inference_server.routes import account, admin, auth, chats, configs, workers
from oasst_inference_server.settings import settings
from oasst_shared.schemas import inference
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.sessions import SessionMiddleware

app = fastapi.FastAPI(title=settings.PROJECT_NAME)


# Allow CORS
app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=settings.inference_cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Session middleware for authlib
app.add_middleware(middleware_class=SessionMiddleware, secret_key=settings.session_middleware_secret_key)


@app.middleware(middleware_type="http")
async def log_exceptions(request: fastapi.Request, call_next):
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Exception in request")
        raise
    return response


# add prometheus metrics at /metrics
@app.on_event(event_type="startup")
async def enable_prom_metrics():
    Instrumentator().instrument(app).expose(app)


@app.on_event(event_type="startup")
async def log_inference_protocol_version():
    logger.warning(f"Inference protocol version: {inference.INFERENCE_PROTOCOL_VERSION}")


def terminate_server(signum, frame):
    logger.warning(f"Signal {signum}. Terminating server...")
    sys.exit(0)


@app.on_event(event_type="startup")
async def alembic_upgrade():
    signal.signal(signal.SIGINT, terminate_server)
    if not settings.update_alembic:
        logger.warning("Skipping alembic upgrade on startup (update_alembic is False)")
        return
    logger.warning("Attempting to upgrade alembic on startup")
    retry = 0
    while True:
        try:
            async with database.make_engine().begin() as conn:
                await conn.run_sync(database.alembic_upgrade)
            logger.warning("Successfully upgraded alembic on startup")
            break
        except Exception:
            logger.exception("Alembic upgrade failed on startup")
            retry += 1
            if retry >= settings.alembic_retries:
                raise

            timeout = settings.alembic_retry_timeout * 2**retry
            logger.warning(f"Retrying alembic upgrade in {timeout} seconds")
            await asyncio.sleep(timeout)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


@app.on_event(event_type="startup")
async def maybe_add_debug_api_keys():
    debug_api_keys = settings.debug_api_keys_list
    if not debug_api_keys:
        logger.warning("No debug API keys configured, skipping")
        return
    try:
        logger.warning("Adding debug API keys")
        async with deps.manual_create_session() as session:
            for api_key in debug_api_keys:
                logger.info(f"Checking if debug API key {api_key} exists")
                if (
                    await session.exec(
                        statement=sqlmodel.select(entity_0=models.DbWorker).where(models.DbWorker.api_key == api_key)
                    )
                ).one_or_none() is None:
                    logger.info(f"Adding debug API key {api_key}")
                    session.add(instance=models.DbWorker(api_key=api_key, name="Debug API Key"))
                    await session.commit()
                else:
                    logger.info(f"Debug API key {api_key} already exists")
        logger.warning("Finished adding debug API keys")
    except Exception:
        logger.exception("Failed to add debug API keys")
        raise


# add routes
app.include_router(router=account.router)
app.include_router(router=auth.router)
app.include_router(router=admin.router)
app.include_router(router=chats.router)
app.include_router(router=workers.router)
app.include_router(router=configs.router)

# mount plugins
for app_prefix, sub_app in plugins.plugin_apps.items():
    app.mount(path=settings.plugins_path_prefix + app_prefix, app=sub_app)


@app.on_event(event_type="startup")
async def welcome_message():
    logger.warning("Inference server started")
    logger.warning("To stop the server, press Ctrl+C")
