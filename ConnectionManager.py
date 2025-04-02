from typing import List

from fastapi import WebSocket

# Store active connections
class ConnectionManager:
    """
    Class that manages all the connection to a particular websocket path
    It serves potentially more than one WebSocket subscriber, basing on the number of connected devices
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Client disconnected: {websocket.client}")

    async def send_data(self, message: str):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to {connection.client}: {e}")
                self.disconnect(connection)
                raise e
