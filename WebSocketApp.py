import asyncio
import datetime
import io
import json
import multiprocessing
import time
from multiprocessing import Process
from threading import Lock
from contextlib import asynccontextmanager
import logging
import httpx
import numpy as np
from matplotlib import pyplot as plt


class ExcludePredictFilter(logging.Filter):
    def filter(self, record):
        return "/predict" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(ExcludePredictFilter())


from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Any, Optional
from zeroconf import ServiceInfo, Zeroconf
import socket

from zeroconf.asyncio import AsyncZeroconf

from Model.Predictor import Predictor, MLPredictor, CNNPredictor
from Server.ConnectionManager import ConnectionManager
from Utility.DataQueueManager import DataQueueManager
from Model.InnerStreamFunctions import log_to_queue, data_processing, dataset_forward_pass_test, \
    embrace_forward_pass_plot, random_prediction_test
from Server.UnityStream import stream_mockup, respond_to_discovery
import logging
import threading

# Biosensor API exe
biosensor_api = "./ShimmerAPI/ShimmerInterface.exe"

manager = ConnectionManager()
data_task: asyncio.Task | None = None

# Create predictor object
MODEL_NAME = "inception_resnet_v2"
CHECKPOINT_PATH = "Model/SavedModels/inception_resnet_v2_whole_100.pt"
DEVICE = "cuda"

KNN_MODEL_PATH = "./MLModels/Saved/knn_WESAD_6F.joblib"
RF_MODEL_PATH = "./MLModels/Saved/rf_WESAD_6F.joblib"

predictor = None

def create_predictor():
    global predictor
    if predictor is None:
        predictor = CNNPredictor(MODEL_NAME, CHECKPOINT_PATH, DEVICE, threshold=True)
# Global parameters
window_seconds = 15
overlap = 0.5


# Pydantic response model
class Prediction(BaseModel):
    probability: float
    label: int

# Input data model for ML prediction endpoint
class WindowRequest(BaseModel):
    window: List[float] = Field(..., min_items=1)
    sampling_rate: Optional[int] = None

# Cleanly cancel & await it on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.stop_event = asyncio.Event()
    mult_man = multiprocessing.Manager()
    app.state.data_manager = DataQueueManager(mult_man.list())
    create_predictor()
    yield
    app.state.stop_event.set()


logging.getLogger("uvicorn.access").disabled = True
# Websocket application listening and sending data stream
websocketApp = FastAPI(lifespan=lifespan)

@websocketApp.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "ok"}

@websocketApp.websocket("/ws/ss")
async def sensor_in_stream(websocket: WebSocket):
    """
    App route that manages the WebSocket connection and communication to the biosensors data streaming.

    :param websocket: WebSocket endpoint server used for the stream
    :return:
    """
    await websocket.accept()
    data_manager = websocketApp.state.data_manager
    stop = websocketApp.state.stop_event
    try:
        while not stop.is_set():
            await asyncio.sleep(0)
            data = await websocket.receive_text()

            parsed_data = json.loads(data)

            heart_rate = parsed_data.get("HeartRate")
            gsr = parsed_data.get("Gsr")
            ppg = parsed_data.get("Ppg")
            sample_rate = parsed_data.get("SampleRate")
            timestamp = datetime.datetime.now()

            obj = {"heart_rate": heart_rate, "gsr": gsr, "ppg": ppg, "sample_rate": sample_rate, "timestamp": timestamp}
            #print("Received data: {}".format(data))

            data_manager.push_single(obj)

    except WebSocketDisconnect as exc:
        print(f"WebSocket disconnected: code={exc.code}")
    except Exception as exc:
        print("WebSocket error:", exc)
    finally:
        print("Sensor connection closed.")




@websocketApp.websocket("/ws/ubs")
async def model_stream(websocket: WebSocket):
    """
    App route that manages the WebSocket connection and communication to the Unity plugin. It launches the preprocess and inference thread
    that produces the result of the classification and sends it to the Unity plugin using the websocket manager.
    Potentially it could stream to multiple Unity applications

    :param websocket: WebSocket endpoint server used for the stream
    :return:
    """
    await manager.connect(websocket)
    data_manager = websocketApp.state.data_manager
    stop = websocketApp.state.stop_event
    stop.clear()
    try:
        def live_mock_predict(*args, **kwargs):
            return mock_slider_value
        await data_processing(data_manager, manager, window_seconds, overlap, stop, live_mock_predict)
    except WebSocketDisconnect:
        print("Application client disconnected, closing connection...\n")
    except Exception as exc:
        print("Model processing stream error:", exc)


# CNN Predict endpoint
@websocketApp.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=415, detail="Unsupported file type")
    image = await file.read()
    try:
        img = Image.open(io.BytesIO(image)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot open image")

    prob, label = await run_in_threadpool(predictor.predict, img)
    return Prediction(probability=prob, label=label)

# ML Predict endpoint
@websocketApp.post("/predict_ml", response_model=Prediction)
async def predict_ml(req: WindowRequest):
    window = np.asarray(req.window, dtype=float).ravel()

    label = await run_in_threadpool(predictor.predict, window)
    return Prediction(probability=0.0, label=label)

def get_service_info(service_name, stream_id, port):
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    service_type = "_ws._tcp.local."
    full_service_name = f"{service_name}.{service_type}"

    properties = {
        "stream": stream_id,  # Stream identifier
        "path": f"/ws/{stream_id}"  # Path for the WebSocket endpoint
    }

    # Return all service information
    info = ServiceInfo(
        type_=service_type,
        name=full_service_name,
        addresses=[socket.inet_aton(local_ip)],
        port=port,
        properties=properties,
        server=f"{hostname}.local."
    )
    return info


async def advertise_service(service_name, stream_id, port):
    """
    Advertise an existant WebSocket server with additional information

    :param service_name: Unique name for the advertised service
    :param stream_id: Identifier for the advertised stream
    :param port: The port where the WebSocket server listens
    """
    info = get_service_info(service_name, stream_id, port)
    zeroconf = AsyncZeroconf()
    await zeroconf.async_register_service(info)
    ips = [socket.inet_ntoa(addr) for addr in info.addresses]
    props = {k.decode(): v.decode() for k, v in info.properties.items()}
    print(f"Advertised {info.name} at {ips} with properties {props}")
    return zeroconf, info

async def launch_process(path, cancel_token):
    async def read_stream(stream, name):
        async for raw in stream:
            line = raw.decode(errors="ignore").rstrip()
            print(f"BiosensorAPI-{name}> {line}")

    proc = await asyncio.create_subprocess_exec(path, "", "", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    # Create terminal output tasks
    stdout_task = asyncio.create_task(read_stream(proc.stdout, "STDOUT"))
    stderr_task = asyncio.create_task(read_stream(proc.stderr, "STDERR"))
    print("Started API process...")

    # Wait to terminate
    proc_wait_task = asyncio.create_task(proc.wait())
    cancel_wait_task = asyncio.create_task(cancel_token.wait())

    done, pending = await asyncio.wait( {proc_wait_task, cancel_wait_task}, return_when=asyncio.FIRST_COMPLETED)
    print("Stopping Biosensor API...")
    if cancel_token.is_set() and proc.returncode is None:
        proc.terminate()
        await proc.wait()

    for task in pending:
        task.cancel()

    await stdout_task
    await stderr_task


async def wait_for_health(url: str, timeout: float = 20.0, interval: float = 0.2):
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return
            except:
                pass
            await asyncio.sleep(interval)
    raise TimeoutError(f"Health‐check failed at {url}")

def run_server():
    uvicorn.run(websocketApp, host="0.0.0.0", port=8000, ws_ping_interval=None)

async def user_input_loop(info_unity):
    respond_to_discovery([info_unity])
    while True:
        key = await asyncio.to_thread(input,
            "Insert h to rediscover, any other key to exit:\n")
        if key == "h":
            respond_to_discovery([info_unity])
        else:
            return

async def main():
    server_process = Process(target=run_server, args=())
    server_process.start()

    # Wait server to effectively run
    await wait_for_health("http://127.0.0.1:8000/health", timeout=35.0)
    threading.Thread(target=start_mock_slider, daemon=True).start()
    sensor_stream = asyncio.create_task(advertise_service("sensor_stream", "ss", 8000))

    # Start API for biosensor
    #api_token = asyncio.Event()
    #api_task = asyncio.create_task(launch_process(biosensor_api, api_token))

    # Start for the first time the discovery of external connection requests to the unity websocket
    await asyncio.sleep(10)
    await random_prediction_test()
    # Test code to maintain the server active
    try:
        info_unity = get_service_info("unitybiosignal_stream", "ubs", 8000)
        await user_input_loop(info_unity)

    finally:
        # Terminate biosensor api
        #api_token.set()
        #await api_task
        # Stop webserver services from being advertised
        zeroconf_sensor, info_sensor = await sensor_stream
        await zeroconf_sensor.async_unregister_service(info_sensor)
        await zeroconf_sensor.async_close()
        # Terminate server process
        print("Stopping WebSocket server...")
        server_process.terminate()
        server_process.join()
        print("Closing connection...")


mock_slider_value = 0

def start_mock_slider():
    import tkinter as tk

    def on_move(v):
        # v is a string, convert to int
        global mock_slider_value
        mock_slider_value = int(v)

    root = tk.Tk()
    root.title("Mock Prediction Slider")
    # slider from 0 to 1
    scale = tk.Scale(root,
                     from_=0, to=1,
                     orient="horizontal",
                     command=on_move,
                     length=300,
                     label="Prediction (0=NotStress,1=Stress)")
    scale.pack(padx=20, pady=20)
    root.mainloop()



if __name__ == "__main__":
    asyncio.run(main())


