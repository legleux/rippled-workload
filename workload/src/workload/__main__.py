import asyncio
import uvicorn
from contextlib import suppress

def main():
    with suppress(KeyboardInterrupt, asyncio.CancelledError):
        uvicorn.run("main.app:app", host="0.0.0.0", port=8000, lifespan="on")
