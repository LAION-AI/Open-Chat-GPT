# A FastAPI server to run the safety pipeline

import asyncio

import fastapi
import uvicorn
from blade2blade import Blade2Blade
from loguru import logger
from oasst_shared.schemas import inference
from settings import settings

app = fastapi.FastAPI()


@app.middleware("http")
async def log_exceptions(request: fastapi.Request, call_next):
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Exception in request")
        raise
    return response


pipeline_loaded: bool = False
pipeline: Blade2Blade


@app.on_event("startup")
async def load_pipeline():
    global pipeline_loaded, pipeline
    pipeline = Blade2Blade(settings.safety_model_name)
    # warmup
    input = "|prompter|Hey,how are you?|endoftext|"
    _ = pipeline.predict(input)
    pipeline_loaded = True
    logger.info("Pipeline loaded")


@app.post("/safety", response_model=inference.SafetyResponse)
async def safety(request: inference.SafetyRequest):
    global pipeline_loaded, pipeline
    while not pipeline_loaded:
        await asyncio.sleep(0.1)
    outputs = pipeline.predict(request.inputs)
    return inference.SafetyResponse(outputs=outputs)


@app.get("/health")
async def health():
    if not pipeline_loaded:
        raise fastapi.HTTPException(status_code=503, detail="Server not fully loaded")
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
