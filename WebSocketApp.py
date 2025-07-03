import asyncio
import datetime
import io
import json
import multiprocessing
from multiprocessing import Process
from threading import Lock
from contextlib import asynccontextmanager

from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
import uvicorn
from pydantic import BaseModel
from zeroconf import ServiceInfo, Zeroconf
import socket

from zeroconf.asyncio import AsyncZeroconf

from Model.Predictor import Predictor
from Server.ConnectionManager import ConnectionManager
from Utility.DataQueueManager import DataQueueManager
from Model.InnerStreamFunctions import log_to_queue, data_processing, dataset_forward_pass_test
from Server.UnityStream import stream_mockup, respond_to_discovery


# Log File Path
log_file = "Log/Stream.txt"
# Biosensor API exe
biosensor_api = "./ShimmerAPI/ShimmerInterface.exe"

manager = ConnectionManager()
data_task: asyncio.Task | None = None

# Create predictor object
MODEL_NAME = "densenet121"
CHECKPOINT_PATH = "./Model/SavedModels/densenet121_differential_100.pt"
DEVICE = "cuda"

predictor = Predictor(MODEL_NAME, CHECKPOINT_PATH, DEVICE)

# Pydantic response model
class Prediction(BaseModel):
    probability: float
    label: int

# Cleanly cancel & await it on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.stop_event = asyncio.Event()
    mult_man = multiprocessing.Manager()
    app.state.data_manager = DataQueueManager(mult_man.list())
    yield
    app.state.stop_event.set()


# Websocket application listening and sending data stream
websocketApp = FastAPI(lifespan=lifespan)
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
    while not stop.is_set():
        data = await websocket.receive_text()
        try:
            parsed_data = json.loads(data)

            heart_rate = parsed_data.get("HeartRate")
            gsr = parsed_data.get("Gsr")
            ppg = parsed_data.get("Ppg")
            sample_rate = parsed_data.get("SampleRate")
            timestamp = datetime.datetime.now()

            obj = {"heart_rate": heart_rate, "gsr": gsr, "ppg": ppg, "sample_rate": sample_rate, "timestamp": timestamp}
            # Log received data to a text file
            data_manager.push_single(obj)

        except WebSocketDisconnect as exc:
            print(f"WebSocket disconnected: code={exc.code}")
        except Exception as exc:
            print("WebSocket error:", exc)
        finally:
            print("Cleaning up sensor_in_stream")



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
        await data_processing(data_manager, manager, stop)
    except WebSocketDisconnect:
        print("Application client disconnected, closing connection...\n")
    finally:
        manager.disconnect(websocket)
    '''
        try:
         await stream_mockup(manager)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        respond_to_discovery()
    '''


# Predict endpoint
@websocketApp.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=415, detail="Unsupported file type")

    image = await file.read()
    try:
        img = Image.open(io.BytesIO(image)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot open image")

    prob, label = predictor.predict(img)
    return Prediction(probability=prob, label=label)

def get_service_info(service_name, stream_id, port):
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    service_type = "_ws._tcp.local."
    full_service_name = f"{service_name}.{service_type}"

    # Add properties such as the stream identifier and the endpoint path
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
    sensor_stream = asyncio.create_task(advertise_service("sensor_stream", "ss", 8000))

    # Start API for biosensor
    api_token = asyncio.Event()
    api_task = asyncio.create_task(launch_process(biosensor_api, api_token))

    # Start for the first time the discovery of external connection requests to the unity websocket
    #dataset_forward_pass_test()
    # Test code to maintain the server active
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
        print("Stopping WebSocket server...")
        server_process.terminate()
        server_process.join()
        print("Closing connection...")

if __name__ == "__main__":
    asyncio.run(main())


