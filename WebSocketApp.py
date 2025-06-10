import asyncio
import json
import multiprocessing
from multiprocessing import Process
from threading import Lock
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from zeroconf import ServiceInfo, Zeroconf
import socket

from Server.ConnectionManager import ConnectionManager
from Utility.DataQueueManager import DataQueueManager
from Model.InnerStreamFunctions import log_to_queue, data_processing
from Server.UnityStream import stream_mockup, respond_to_discovery


# Log File Path
log_file = "Log/Stream.txt"

manager = ConnectionManager()
data_task: asyncio.Task | None = None


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

            obj = {"heart_rate": heart_rate, "gsr": gsr, "ppg": ppg, "sample_rate": sample_rate}
            # Log received data to a text file
            data_manager.push_single(obj)


        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
        except WebSocketDisconnect as e:
            print(f"Sensor streaming client disconnected {e}, closing connection...")



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




def advertise_service(service_name: str, stream_id: str, port: int):
    """
    Advertise an existant WebSocket server with additional information

    :param service_name: Unique name for the advertised service
    :param stream_id: Identifier for the advertised stream
    :param port: The port where the WebSocket server listens
    """
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

    zeroconf = Zeroconf()
    zeroconf.register_service(info)
    print(f"Service '{full_service_name}' advertised at {local_ip}:{port} with properties {properties}")
    return zeroconf, info

def run_server():
    uvicorn.run(websocketApp, host="0.0.0.0", port=8000, ws_ping_interval=None)


if __name__ == "__main__":
    server_process = Process(target=run_server, args=())
    server_process.start()
    zeroconf_sensor, info_sensor = advertise_service("sensor_stream", "ss", 8000)
    zeroconf_unity, info_unity = advertise_service("unitybiosignal_stream", "ubs", 8000)
    # Start for the first time the discovery of external connection requests to the unity websocket
    respond_to_discovery([info_unity])
    # Test code to maintain the server active
    try:
        while True:
            key = input("Insert h to start again the discovery\nany other key to exit...\n\n")
            if key == "h":
                respond_to_discovery([info_unity])
            else:
                break

    finally:
        zeroconf_sensor.unregister_service(info_sensor)
        zeroconf_sensor.close()
        zeroconf_unity.unregister_service(info_unity)
        zeroconf_unity.close()
        # Terminate server process
        print("Stopping WebSocket server...")
        server_process.terminate()
        server_process.join()
        print("Closing connection...")



