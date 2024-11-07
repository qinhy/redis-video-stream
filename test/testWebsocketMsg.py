import asyncio
import json
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

shared_data = {"count": 0}

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    html_content = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebSocket Client</title>
</head>
<body>
    <h1>WebSocket Client</h1>
    <input type="text" id="messageInput" placeholder="Type a message...">
    <button onclick="sendMessage()">Send</button>
    <ul id="messages"></ul>

    <script>
        // Create a new WebSocket connection
        const websocket = new WebSocket('ws://localhost:8000/ws');

        // Connection opened event
        websocket.onopen = function () {
            console.log('Connected to WebSocket server');
        };

        // Connection closed event
        websocket.onclose = function () {
            console.log('Disconnected from WebSocket server');
        };

        // Message received from server
        websocket.onmessage = function (event) {
            const messageList = document.getElementById('messages');
            const newMessage = document.createElement('li');
            newMessage.textContent = `Server: ${event.data}`;
            messageList.appendChild(newMessage);
        };

        // Error event
        websocket.onerror = function (error) {
            console.error('WebSocket Error:', error);
        };

        // Send a message to the server
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value;
            if (message) {
                websocket.send(message);  // Send message to server
                input.value = '';  // Clear input
            }
        }
    </script>
</body>
</html>
    """
    return html_content

@app.get("/count")
async def count():
    return JSONResponse(content={"count": shared_data["count"]})

# Connect to Redis using redis-py
redis_client = redis.Redis(host='localhost', decode_responses=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pubsub = redis_client.pubsub()
    shared_data["count"] += 1
    await pubsub.subscribe("chat_channel")  # Subscribe to a Redis channel
    
    async def send_messages():
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"])
    
    # Create a background task for sending messages
    send_task = asyncio.create_task(send_messages())
    
    try:
        while True:
            data = await websocket.receive_text()
            await redis_client.publish("chat_channel", data)  # Publish messages to Redis
    except Exception as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        shared_data["count"] -= 1
        send_task.cancel()  # Clean up the background task when the connection is closed
        await pubsub.unsubscribe("chat_channel")
        # await websocket.close()