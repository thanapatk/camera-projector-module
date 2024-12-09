from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from multiprocessing import Queue
import time

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://thanapatk.local:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartProjectorItem(BaseModel):
    calibrate: bool


class UploadImageItem(BaseModel):
    image: str


def run_fastapi(projector_started, to_ws: Queue, from_ws: Queue):
    @app.post("/start_projector")
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

    @app.get("/projector_started")
    def is_projector_started():
        return {"started": projector_started.value}

    @app.post("/upload_image")
    def upload_image(item: UploadImageItem):
        from_ws.put(f"upload_image {item.image}")

        while True:
            if not to_ws.empty():
                break

            time.sleep(0.5)

        if to_ws.get() != "uploaded_image":
            raise HTTPException(status_code=500, detail="Upload Image Failed.")

        return {"message": "ok"}

    async def send_to_websocket(websocket: WebSocket, queue: Queue):
        while True:
            if not queue.empty():
                data = queue.get()
                print(data)
                await websocket.send_text(data)

            await asyncio.sleep(0.5)

    async def receive_from_websocket(websocket: WebSocket, queue: Queue):
        while True:
            data = await websocket.receive_text()
            queue.put(data)

    @app.websocket("/project_ws")
    async def project_ws(websocket: WebSocket):
        await websocket.accept()
        try:
            # Create tasks for sending and receiving data concurrently
            send_task = asyncio.ensure_future(send_to_websocket(websocket, to_ws))
            receive_task = asyncio.ensure_future(
                receive_from_websocket(websocket, from_ws)
            )

            # Wait for either task to complete (e.g., disconnection)
            done, pending = await asyncio.wait(
                [send_task, receive_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel any pending tasks
            for task in pending:
                task.cancel()
        except WebSocketDisconnect:
            print("WebSocket disconnected.")
        except Exception as e:
            print(f"WebSocket error: {e}")

    __import__("uvicorn").run(app, host="0.0.0.0", port=8000)
