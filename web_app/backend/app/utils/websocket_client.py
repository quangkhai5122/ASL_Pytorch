"""
WebSocket client utilities and helpers for testing and integration.
Provides examples of how to connect to and use the WebSocket API.
"""

import asyncio
import json
import base64
import websockets
from typing import Optional, Callable
from pathlib import Path


class WebSocketClient:
    """
    WebSocket client for real-time sign recognition.
    Handles connection, frame streaming, and message parsing.
    """

    def __init__(
        self,
        api_url: str = "ws://localhost:8000",
        api_token: str = None,
        on_prediction: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        """
        Initialize WebSocket client.

        Args:
            api_url: WebSocket server URL
            api_token: JWT authentication token
            on_prediction: Callback for predictions
            on_error: Callback for errors
        """
        self.api_url = api_url.rstrip("/")
        self.api_token = api_token
        self.websocket = None
        self.connected = False
        self.on_prediction = on_prediction
        self.on_error = on_error
        self.frame_id = 0

    async def connect(self) -> bool:
        """
        Establish WebSocket connection.

        Returns:
            bool: True if connected successfully
        """
        if not self.api_token:
            print("Error: API token required")
            return False

        try:
            ws_url = f"{self.api_url}/api/v1/ws/stream?token={self.api_token}"
            self.websocket = await websockets.connect(ws_url)
            self.connected = True
            print(f"✓ Connected to {ws_url}")

            # Receive welcome message
            welcome = await self.websocket.recv()
            print(f"✓ Server: {welcome}")

            return True

        except Exception as e:
            print(f"✗ Connection failed: {str(e)}")
            self.connected = False
            return False

    async def disconnect(self):
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.send(json.dumps({"type": "close"}))
            await self.websocket.close()
            self.connected = False
            print("✓ Disconnected")

    async def send_frame(self, frame_data: bytes, frame_id: int = None) -> bool:
        """
        Send frame for prediction.

        Args:
            frame_data: JPEG frame bytes
            frame_id: Optional frame ID

        Returns:
            bool: True if sent successfully
        """
        if not self.connected:
            print("Error: Not connected")
            return False

        if frame_id is None:
            self.frame_id += 1
            frame_id = self.frame_id

        try:
            # Encode frame to base64
            frame_base64 = base64.b64encode(frame_data).decode("utf-8")

            # Send message
            message = {
                "type": "frame",
                "data": {
                    "frame_base64": frame_base64,
                    "frame_id": frame_id,
                },
            }

            await self.websocket.send(json.dumps(message))
            return True

        except Exception as e:
            print(f"Error sending frame: {str(e)}")
            return False

    async def send_frame_from_file(self, file_path: str) -> bool:
        """
        Send frame from JPEG file.

        Args:
            file_path: Path to JPEG file

        Returns:
            bool: True if sent successfully
        """
        if not Path(file_path).exists():
            print(f"File not found: {file_path}")
            return False

        with open(file_path, "rb") as f:
            frame_data = f.read()

        return await self.send_frame(frame_data)

    async def receive_messages(self):
        """
        Receive and handle messages from server.
        Runs until connection closed.
        """
        if not self.connected:
            print("Error: Not connected")
            return

        try:
            while self.connected:
                message_str = await self.websocket.recv()
                message = json.loads(message_str)

                # Handle different message types
                msg_type = message.get("type")

                if msg_type == "prediction":
                    if self.on_prediction:
                        self.on_prediction(message)
                    else:
                        print(f"Prediction: {message.get('sign')} ({message.get('confidence')})")

                elif msg_type == "heartbeat":
                    uptime = message.get("uptime_seconds")
                    frames = message.get("frames_received")
                    preds = message.get("predictions_made")
                    latency = message.get("avg_latency_ms")
                    print(
                        f"♥ Heartbeat: {frames} frames, {preds} predictions, {latency}ms avg"
                    )

                elif msg_type == "error":
                    error = message.get("error")
                    print(f"✗ Error: {error}")
                    if self.on_error:
                        self.on_error(message)

                elif msg_type == "frame_received":
                    print(f"Frame {message.get('frame_id')} received")

                else:
                    print(f"Message: {message}")

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed by server")
            self.connected = False

        except Exception as e:
            print(f"Error receiving message: {str(e)}")
            self.connected = False

    async def request_status(self) -> Optional[dict]:
        """
        Request connection status from server.

        Returns:
            Status dict or None if error
        """
        if not self.connected:
            return None

        try:
            message = {"type": "status"}
            await self.websocket.send(json.dumps(message))

            # Wait for response
            response_str = await asyncio.wait_for(self.websocket.recv(), timeout=5)
            response = json.loads(response_str)

            if response.get("type") == "status":
                return response.get("data")

            return None

        except Exception as e:
            print(f"Error requesting status: {str(e)}")
            return None

    async def ping(self) -> bool:
        """
        Send ping to verify connection.

        Returns:
            bool: True if ping/pong successful
        """
        if not self.connected:
            return False

        try:
            import time

            message = {
                "type": "ping",
                "timestamp": time.time(),
            }

            await self.websocket.send(json.dumps(message))

            # Wait for pong
            response_str = await asyncio.wait_for(self.websocket.recv(), timeout=5)
            response = json.loads(response_str)

            if response.get("type") == "pong":
                return True

            return False

        except Exception as e:
            print(f"Ping failed: {str(e)}")
            return False


# Example usage
async def example_stream():
    """
    Example: Stream frames from webcam (requires OpenCV).
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return

    # Initialize client
    client = WebSocketClient(
        api_url="ws://localhost:8000",
        api_token="YOUR_JWT_TOKEN_HERE",
    )

    # Connect
    if not await client.connect():
        return

    # Capture from webcam
    cap = cv2.VideoCapture(0)

    try:
        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Encode frame
            _, frame_data = cv2.imencode(".jpg", frame)

            # Send frame
            await client.send_frame(frame_data.tobytes())

            # Check for predictions
            frame_count += 1

            if frame_count % 30 == 0:  # Every 30 frames
                status = await client.request_status()
                if status:
                    print(f"Status: {status['total_frames']} frames processed")

            # Press 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        await client.disconnect()


# Python client test script
if __name__ == "__main__":
    import asyncio

    async def main():
        # This requires API token - get from login endpoint first
        print("WebSocket Client Example")
        print("-" * 40)
        print("To use: export your JWT token and modify the script")

        # Example of connecting and receiving messages
        # client = WebSocketClient(
        #     api_url="ws://localhost:8000",
        #     api_token="YOUR_TOKEN",
        # )
        #
        # if await client.connect():
        #     # Start receiving task
        #     receive_task = asyncio.create_task(client.receive_messages())
        #     
        #     # Send test frames every second
        #     for i in range(10):
        #         await client.send_frame(b"test_frame_data", frame_id=i)
        #         await asyncio.sleep(1)
        #     
        #     await client.disconnect()

    asyncio.run(main())
