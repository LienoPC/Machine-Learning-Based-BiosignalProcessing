import asyncio
import json
import socket


# Port on which UDP messages for discovery are received
DISCOVERY_PORT = 37020
# Port on which WebSocket server is running on
SERVER_WS_PORT = 8000
SERVICE_INFO = {
    "ws_port": SERVER_WS_PORT,
    "path": "/ws/ubs",     # WebSocket endpoint path
    "stream": "ubs",       # Stream identifier
}


def respond_to_discovery():
    """
    Function that listens for UDP discovery packets (sent ideally by the unity application)
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', DISCOVERY_PORT))
    print(f"Discovery server listening on UDP port {DISCOVERY_PORT}")

    while True:
        data, addr = sock.recvfrom(1024)
        message = data.decode('utf-8')
        print(f"Received discovery message: {message} from {addr}")
        if message.strip() == "DISCOVER_WEBSOCKET":
            # Serialize the entire service info as JSON
            response = json.dumps(SERVICE_INFO)
            sock.sendto(response.encode('utf-8'), addr)
            print(f"Responded to {addr} with: {response}")

async def stream_packet(connection_manager, packet):
    await connection_manager.send_data(json.dumps(packet))
    print(f"Sent: {packet}")


async def stream_mockup(websocket):
        """

        :param websocket: connection manager for the created websocket

        """
        while True:
            try:
                data = {
                    "heart_rate": 72,
                    "gsr": 0.5,
                    "psr": 1.2
                }
                await stream_packet(websocket, data)
                await asyncio.sleep(0.05)

            except Exception as e:
                print("Error while sending data:", e)
                break



