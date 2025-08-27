import asyncio
import datetime
import io
import json
import multiprocessing
from multiprocessing import Process
from contextlib import asynccontextmanager
import logging
import httpx
import numpy as np

from Utility.ModelTestProcessFunctions import random_prediction_test


# Disables predict endpoint logging
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
from Model.InnerStreamFunctions import data_processing
from Server.UnityStream import stream_mockup, respond_to_discovery
import logging
import threading

# Biosensor API exe
BIOSENSOR_API = "./ShimmerAPI/ShimmerInterface.exe"

# Start connection manager and define data task object
manager = ConnectionManager()
data_task: asyncio.Task | None = None

# Create predictor object
MODEL_NAME = "resnet50"
CHECKPOINT_PATH = "Model/SavedModels/resnet50_whole_100.pt"
DEVICE = "cuda"

KNN_MODEL_PATH = "./MLModels/Saved/knn_WESAD_6F.joblib"
RF_MODEL_PATH = "./MLModels/Saved/rf_WESAD_6F.joblib"

# Global parameters
window_seconds = 15
overlap = 0.2
predictor = None

shared_slider = None
def create_predictor():
    """
    Creates predictor object. Called once at startup
    :return:
    """
    global predictor
    if predictor is None:
        predictor = CNNPredictor(MODEL_NAME, CHECKPOINT_PATH, DEVICE, threshold=True)



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
    Manages data received by websocket and push it inside a shared data list

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
            data_manager.push_single(obj)

    except WebSocketDisconnect as exc:
        print(f"SERVER/WS/SS: WebSocket disconnected: code={exc.code}")
    except Exception as exc:
        print("SERVER/WS/SS: WebSocket error:", exc)
    finally:
        print("SERVER/WS/SS: Sensor connection closed.")


@websocketApp.websocket("/ws/ubs")
async def model_stream(websocket: WebSocket):
    """
    App route that manages the WebSocket connection and communication with the Unity plugin. It launches the preprocess and inference thread
    that produces the result of the classification and sends it to the Unity plugin using the websocket manager.
    It could potentially stream to multiple Unity applications

    :param websocket: WebSocket endpoint server used for the stream
    :return:
    """
    await manager.connect(websocket)
    data_manager = websocketApp.state.data_manager
    stop = websocketApp.state.stop_event
    stop.clear()
    try:
        await data_processing(data_manager, manager, window_seconds, overlap, 4, stop)
    except WebSocketDisconnect:
        print("SERVER/WS/UBS: Application client disconnected, closing connection...\n")
    except Exception as exc:
        print("SERVER/WS/UBS: Model processing stream error:", exc)


# CNN Predict endpoint
@websocketApp.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint route. Receives an encoded image as File and feeds it to the predictor

    :param file: encoded image file
    :return: Prediction object that contains the label produced and probability computed by the model
    """
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
    """
    Creates ServiceInfo object to be advertised to potential clients
    :param service_name:
    :param stream_id:
    :param port:
    :return:
    """
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
    Advertise an existent WebSocket server with additional information

    :param service_name: Unique name for the advertised service
    :param stream_id: Identifier for the advertised stream
    :param port: The port where the WebSocket server listens
    """
    info = get_service_info(service_name, stream_id, port)
    zeroconf = AsyncZeroconf()
    await zeroconf.async_register_service(info)
    ips = [socket.inet_ntoa(addr) for addr in info.addresses]
    props = {k.decode(): v.decode() for k, v in info.properties.items()}
    print(f"SERVER: Advertised {info.name} at {ips} with properties {props}")
    return zeroconf, info


async def launch_process(path, cancel_token, *args):
    """
    Starts a secondary process and prepares a secondary thread to print stdout and stderr of the process. Sets the cancel token when the process terminates
    :param path: executable file path
    :param cancel_token: cancel token that will be set when the process ends
    :return:
    """
    async def read_stream(stream, name):
        async for raw in stream:
            line = raw.decode(errors="ignore").rstrip()
            print(f"API: BiosensorAPI-{name}> {line}")

    proc = await asyncio.create_subprocess_exec(path, *args, "", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    # Create terminal output tasks
    stdout_task = asyncio.create_task(read_stream(proc.stdout, "STDOUT"))
    stderr_task = asyncio.create_task(read_stream(proc.stderr, "STDERR"))
    print("API: Started API process...")

    # Wait to terminate
    proc_wait_task = asyncio.create_task(proc.wait())
    cancel_wait_task = asyncio.create_task(cancel_token.wait())

    done, pending = await asyncio.wait( {proc_wait_task, cancel_wait_task}, return_when=asyncio.FIRST_COMPLETED)
    print("API: Stopping Biosensor API...")
    if cancel_token.is_set() and proc.returncode is None:
        proc.terminate()
        await proc.wait()

    for task in pending:
        task.cancel()

    # Join read stream threads
    await stdout_task
    await stderr_task


async def wait_for_health(url: str, timeout: float = 20.0, interval: float = 0.2):
    """
    Function executed at server startup, waits for all the server endpoint to be available
    :param url:
    :param timeout:
    :param interval:
    :return:
    """
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
    raise TimeoutError(f"SERVER: Health‐check failed at {url}")


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
    # TODO: Remove, testing only
    #manager_proc = multiprocessing.Manager()
    #slider_shared = manager_proc.Value('i', 0)

    server_process = Process(target=run_server, args=())
    server_process.start()

    # Wait server to effectively run
    await wait_for_health("http://127.0.0.1:8000/health", timeout=20.0)
    #threading.Thread(target=start_mock_slider, args=(slider_shared,), daemon=True).start() # TODO: Remove, testing only
    sensor_stream = asyncio.create_task(advertise_service("sensor_stream", "ss", 8000))

    # Start API for biosensor
    api_token = asyncio.Event()
    api_task = asyncio.create_task(launch_process(BIOSENSOR_API, api_token, "COM4", "32"))

    #await asyncio.sleep(10)
    #await random_prediction_test()
    # Start user input loop
    try:
        info_unity = get_service_info("unitybiosignal_stream", "ubs", 8000)
        await user_input_loop(info_unity)

    finally:
        # Terminate biosensor api
        api_token.set()
        await api_task
        # Stop webserver services from being advertised
        zeroconf_sensor, info_sensor = await sensor_stream
        await zeroconf_sensor.async_unregister_service(info_sensor)
        await zeroconf_sensor.async_close()
        # Terminate server process
        print("SERVER: Stopping WebSocket endpoints...")
        server_process.terminate()
        server_process.join()
        print("SERVER: Closing connection...")


def start_mock_slider(shared_proxy):
    import tkinter as tk

    def on_move(v):
        print(f"Setting mock slider value to: {v}")
        shared_proxy.value = int(v)

    root = tk.Tk()
    root.title("Mock Prediction Slider")
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


