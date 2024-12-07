from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
import asyncio
import time

app = FastAPI()


class StartProjectorItem(BaseModel):
    calibrate: bool


def run_fastapi(projector_started, to_ws, from_ws):
    @app.post("/start-projector")
    def start_projector(item: StartProjectorItem):
        calibrate = item.calibrate

        from_ws.put(f"start_projector_{'with' if calibrate else 'no'}_calibration")

        while True:
            if not to_ws.empty():
                break

            time.sleep(0.5)

        if to_ws.get() != "started_projector":
            raise HTTPException(status_code=500, detail="Projector failed to start.")

        return {"message": "ok"}

    @app.get("/projector-started")
    def is_projector_started():
        return {"started": projector_started.value}

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
