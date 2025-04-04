import asyncio
import json
import socket


# Port on which UDP messages for discovery are received
DISCOVERY_PORT = 37020
# Port on which WebSocket server is running on
SERVER_WS_PORT = 8000

def respond_to_discovery(service_info, timeout=20):
    """
    Listens for UDP discovery packets and responds with SERVICE_INFO.
    The socket will wait for a discovery message until the timeout elapses.

    :param timeout: Time (in seconds) to wait for a packet before giving up.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', DISCOVERY_PORT))
    sock.settimeout(timeout)  # Set the timeout on the socket
    print(f"Discovery server listening on UDP port {DISCOVERY_PORT} with a timeout of {timeout} seconds.")

    while True:
        try:
            # This call will block until data is received or timeout occurs.
            data, addr = sock.recvfrom(1024)
        except socket.timeout:
            print(f"Exiting discovery loop...")
            break  # Exit the loop if no data is received within the timeout period

        message = data.decode('utf-8')
        print(f"Received discovery message: {message} from {addr}")
        if message.strip() == "DISCOVER_WEBSOCKET":

            for info in service_info:
                response = json.dumps(info)
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
                    "Test Data": 50,
                }
                await stream_packet(websocket, data)
                await asyncio.sleep(0.05)

            except Exception as e:
                print("Error while sending data:", e)
                break



