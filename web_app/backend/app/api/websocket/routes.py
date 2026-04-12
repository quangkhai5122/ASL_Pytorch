"""
WebSocket endpoint router for real-time streaming.
Handles WebSocket connections and real-time sign recognition.
"""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException, status
from fastapi.security import HTTPBearer

from app.config import settings
from app.core.auth import decode_token
from app.api.websocket.stream import connection_manager

router = APIRouter(prefix="/api/v1", tags=["websocket"])


async def get_token_from_query(token: str = Query(...)) -> str:
    """
    Extract and validate JWT token from query parameter.

    Args:
        token: JWT token from query parameter

    Returns:
        Validated username

    Raises:
        HTTPException: If token is invalid
    """
    username = decode_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return username


@router.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    token: str = Query(...),
):
    """
    WebSocket endpoint for real-time frame streaming and live sign recognition.

    **Connection Flow:**
    1. Client connects with JWT token in query: `ws://localhost:8000/api/v1/ws/stream?token=YOUR_JWT`
    2. Server validates token
    3. Client sends frames in JSON format every ~100ms
    4. Server returns predictions in real-time
    5. Server sends heartbeats every 30 seconds
    6. Connection auto-closes on token expiry or inactivity

    **Message Protocol:**

    **Client → Server (Frame):**
    ```json
    {
        "type": "frame",
        "data": {
            "frame_base64": "/9j/4AAQSkZJRg...",  // JPEG base64
            "frame_id": 1,
            "timestamp": "2024-01-15T10:30:45.123Z"
        }
    }
    ```

    **Server → Client (Prediction):**
    ```json
    {
        "type": "prediction",
        "sign": "hello",
        "confidence": 0.95,
        "top5": [
            {"sign": "hello", "confidence": 0.95},
            {"sign": "hi", "confidence": 0.03}
        ],
        "frame_id": 1,
        "processing_time_ms": 45.2
    }
    ```

    **Server → Client (Heartbeat):**
    ```json
    {
        "type": "heartbeat",
        "timestamp": "2024-01-15T10:30:45.123Z",
        "uptime_seconds": 123.45,
        "frames_received": 45,
        "predictions_made": 10,
        "avg_latency_ms": 42.5
    }
    ```

    **Server → Client (Error):**
    ```json
    {
        "type": "error",
        "error": "Invalid frame format",
        "frame_id": 1
    }
    ```

    Args:
        websocket: WebSocket connection
        token: JWT token from query parameter

    Raises:
        HTTPException: If token is invalid
    """

    # Validate token
    username = decode_token(token)
    if username is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
        return

    # Accept connection (use the returned connection object directly)
    try:
        connection = await connection_manager.connect(websocket, username)
        print(f"[WS] Connection accepted for {username}")
    except Exception as e:
        print(f"[WS] Connection error for {username}: {str(e)}")
        return

    # Start heartbeat task
    heartbeat_task = None

    try:
        # Send welcome message
        welcome_msg = {
            "type": "connected",
            "message": f"Welcome {username}! Ready to receive frames.",
            "server_version": settings.PROJECT_VERSION,
            "max_frames": settings.WINDOW_SIZE,
        }
        await connection.send_json(welcome_msg)
        print(f"[WS] Welcome message sent to {username}")

        # Start heartbeat coroutine
        async def heartbeat_loop():
            """Send periodic heartbeats."""
            try:
                while True:
                    await asyncio.sleep(30)  # Every 30 seconds
                    try:
                        heartbeat_msg = {
                            "type": "heartbeat",
                            "timestamp": datetime.now().isoformat(),
                            "uptime_seconds": (datetime.now() - connection.connected_at).total_seconds(),
                            "frames_received": connection.frame_count,
                            "predictions_made": connection.prediction_count,
                        }
                        await connection.send_json(heartbeat_msg)
                    except Exception as e:
                        print(f"[WS] Heartbeat error for {username}: {str(e)}")
                        break
            except asyncio.CancelledError:
                pass

        heartbeat_task = asyncio.create_task(heartbeat_loop())

        # Main message loop
        while True:
            # Receive message
            message = await connection.receive_frame_message()

            if message is None:
                continue

            # Handle different message types
            if message.get("type") == "frame":
                frame_data = message.get("data", {})
                frame_base64 = frame_data.get("frame_base64")
                frame_id = frame_data.get("frame_id", connection.frame_count)

                if not frame_base64:
                    await connection.send_json(
                        {
                            "type": "error",
                            "error": "Missing frame_base64 in data",
                            "frame_id": frame_id,
                        }
                    )
                    continue

                # Process frame
                result = await connection.process_frame(frame_base64, frame_id)

                if result:
                    await connection.send_json(result)

            elif message.get("type") == "ping":
                # Respond to ping
                await connection.send_json(
                    {
                        "type": "pong",
                        "timestamp": message.get("timestamp"),
                    }
                )

            elif message.get("type") == "status":
                # Send connection stats
                stats = connection.get_stats()
                await connection.send_json(
                    {
                        "type": "status",
                        "data": stats,
                    }
                )

            elif message.get("type") == "close":
                # Client requested connection close
                await websocket.close(code=status.WS_1000_NORMAL_CLOSURE)
                break

            else:
                await connection.send_json(
                    {
                        "type": "error",
                        "error": f"Unknown message type: {message.get('type')}",
                    }
                )

    except WebSocketDisconnect as e:
        print(f"[WS] Client {username} disconnected: {e.code}")
        try:
            stats = connection.get_stats()
            print(f"[WS]   Final stats: {stats['total_frames']} frames, {stats['total_predictions']} predictions")
        except Exception:
            pass

    except Exception as e:
        import traceback
        print(f"[WS] Error for {username}: {str(e)}")
        traceback.print_exc()
        try:
            await websocket.close(code=status.WS_1011_SERVER_ERROR, reason=str(e)[:120])
        except:
            pass

    finally:
        # Cancel heartbeat task
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        # Only remove from manager if this connection is still the active one
        # (prevents a reconnected session from being cleaned up by the old handler)
        current = connection_manager.get_connection(username)
        if current is connection:
            await connection_manager.disconnect(username)

        print(f"[WS] Connection closed for {username}")


@router.get("/ws/stats")
async def websocket_stats():
    """
    Get WebSocket connection statistics.

    Returns:
        Statistics about active WebSocket connections
    """
    active_count = connection_manager.get_active_count()
    connections_data = {}

    for client_id, connection in connection_manager.active_connections.items():
        connections_data[client_id] = connection.get_stats()

    return {
        "active_connections": active_count,
        "connections": connections_data,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }


@router.get("/ws/health")
async def websocket_health():
    """
    Get WebSocket server health status.

    Returns:
        Health information
    """
    return {
        "status": "healthy",
        "active_connections": connection_manager.get_active_count(),
        "max_connections": "unlimited",
        "heartbeat_interval": "30s",
    }
