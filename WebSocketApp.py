import json
from multiprocessing import Process

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from zeroconf import ServiceInfo, Zeroconf
import socket

from ConnectionManager import ConnectionManager
from InnerStreamFunctions import log_to_file
from UnityStream import stream_mockup, respond_to_discovery

# Websocket application listening and sending data stream
websocketApp = FastAPI()

# Log File Path
log_file = "Log/Stream.txt"

manager = ConnectionManager()

@websocketApp.websocket("/ws/ss")
async def sensor_stream(websocket: WebSocket):
    """
    App route that manages the WebSocket connection and communication to the biosensors data streaming.

    :param websocket: WebSocket endpoint server used for the stream
    :return:
    """
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        print(f"Raw received data: {data}")

        # Deserialize JSON
        try:
            parsed_data = json.loads(data)

            heart_rate = parsed_data.get("HeartRate")
            gsr = parsed_data.get("Gsr")
            psr = parsed_data.get("Ppg")

            obj = {heart_rate: heart_rate, gsr: gsr, psr: psr}

            log_to_file(obj, log_file)
            print(f"Received object: {obj}")

        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")



@websocketApp.websocket("/ws/ubs")
async def sensor_stream(websocket: WebSocket):
    """
    App route that manages the WebSocket connection and communication to the Unity plugin.

    :param websocket: WebSocket endpoint server used for the stream
    :return:
    """
    await manager.connect(websocket)
    try:
         await stream_mockup(manager)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        respond_to_discovery()



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
    uvicorn.run(websocketApp, host="0.0.0.0", port=8000)



if __name__ == "__main__":
    server_process = Process(target=run_server)
    server_process.start()

    zeroconf_sensor, info_sensor = advertise_service("sensor_stream", "ss", 8000)
    zeroconf_unity, info_unity = advertise_service("unitybiosignal_stream", "ubs", 8000)

    # Start for the first time the discovery of external connection requests to the unity websocket
    respond_to_discovery()
    # Test code to mantain the server active
    try:
        input("Press enter to exit...\n\n")
    finally:
        zeroconf_sensor.unregister_service(info_sensor)
        zeroconf_sensor.close()
        zeroconf_unity.unregister_service(info_unity)
        zeroconf_unity.close()


