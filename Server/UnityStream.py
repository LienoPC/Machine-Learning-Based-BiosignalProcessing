import asyncio
import json
import random
import socket

from zeroconf import ServiceInfo

# Port on which UDP messages for discovery are received
DISCOVERY_PORT = 37020
# Port on which WebSocket server is running on
SERVER_WS_PORT = 8000

def service_info_to_dict(info: ServiceInfo) -> dict:
    return {
        "stream":     info.decoded_properties.get("stream"),
        "path":       info.decoded_properties.get("path"),
        "ws_port":    info.port,
    }


def respond_to_discovery(service_info, timeout=20):
    """
    Listens for UDP discovery packets and responds with SERVICE_INFO.
    The socket will wait for a discovery message until the timeout elapses.

    :param timeout: Time (in seconds) to wait for a packet before giving up.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', DISCOVERY_PORT))
    sock.settimeout(timeout)  # Set the timeout on the socket

    while True:
        try:
            # This call will block until data is received or timeout occurs.
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            print(f"Exiting discovery loop...")
            break  # Exit the loop if no data is received within the timeout period

        message = data.decode('utf-8')
        if message.strip() == "DISCOVER_WEBSOCKET":

            for info in service_info:
                response = json.dumps(service_info_to_dict(info))
                sock.sendto(response.encode('utf-8'), addr)
                print(f"Responded to {addr} with: {response}")


async def stream_packet(connection_manager, packet):
    if connection_manager.connection_available():
        #print(f"Streaming data: {packet} to VR App")
        await connection_manager.send_data(packet)
        #print(f"Sent: {packet}")


async def unity_stream(result, connection_manager):
    '''

    :param result: Its value can be 0 or 1, 0 means "not stressed" and 1 means "stressed"
    :return:
    '''
    if result == 1:
        data = {
            "Emotion": "Stress",
        }
    else:
        data = {
            "Emotion": "NotStress",
        }
    await stream_packet(connection_manager, data)

async def stream_mockup(websocket):
        """

        :param websocket: connection manager for the created websocket

        """
        while True:
            try:
                range = random.uniform(0,1)

                if range > 0.5:
                    data = {
                        "Emotion": "Stress",
                    }
                else:
                    data = {
                        "Emotion": "NotStress",
                    }

                await stream_packet(websocket, data)
                await asyncio.sleep(1)

            except Exception as e:
                print("Error while sending data:", e)
                break



