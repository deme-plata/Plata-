import asyncio
import websockets
import json
import logging
from typing import Dict, Set, Callable

logger = logging.getLogger(__name__)

class ExchangeWebSocketUpdates:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self.message_handlers: Dict[str, Callable] = {}

    async def start(self):
        self.server = await websockets.serve(self.handle_client, self.host, self.port)
        logger.info(f"WebSocket server started on {self.host}:{self.port}")

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        finally:
            self.clients.remove(websocket)

    async def process_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        try:
            data = json.loads(message)
            message_type = data.get('type')
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](websocket, data)
            else:
                logger.warning(f"Unhandled message type: {message_type}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

    def register_handler(self, message_type: str, handler: Callable):
        self.message_handlers[message_type] = handler

    async def broadcast(self, message: dict):
        if not self.clients:
            return
        ws_message = json.dumps(message)
        await asyncio.gather(
            *[client.send(ws_message) for client in self.clients]
        )

    async def send_to_client(self, websocket: websockets.WebSocketServerProtocol, message: dict):
        await websocket.send(json.dumps(message))