import asyncio
import json
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

shared_data = {"count": 0}

redis_client = redis.Redis(host='localhost', decode_responses=True)  # Redis client connection


@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebSocket Client</title>
</head>
<body>
    <h1>WebSocket Client</h1>
    <div>
        <label for="channel">Channel: </label>
        <input type="text" id="channel" placeholder="Enter channel name">
        <button id="connectPubSub">Connect to PubSub</button>
        <button id="connectCommand">Connect to Command WS</button>
    </div>
    
    <div id="pubsubControls" style="display:none;">
        <h3>PubSub Messages</h3>
        <input type="text" id="pubsubMessage" placeholder="Enter message">
        <button id="sendPubSub">Send to Channel</button>
        <div id="pubsubOutput"></div>
    </div>

    <div id="commandControls" style="display:none;">
        <h2>WebSocket Commands (SET, GET, DEL, KEYS)</h2>
        <select id="command">
            <option value="SET">SET</option>
            <option value="GET">GET</option>
            <option value="DEL">DEL</option>
            <option value="KEYS">KEYS</option>
        </select>
        <input type="text" id="key" placeholder="Key">
        <input type="text" id="value" placeholder="Value (for SET)">
        <button id="sendCommand">Send Command</button>
        <div id="generalOutput"></div>
    </div>

    <script>
        let wsPubSub;
        let wsCommand;

        document.getElementById('connectPubSub').addEventListener('click', function() {
            const channel = document.getElementById('channel').value;
            if (!channel) {
                alert("Please enter a channel name.");
                return;
            }

            wsPubSub = new WebSocket(`ws://localhost:8000/ws/pubsub/${channel}`);
            wsPubSub.onopen = function() {
                document.getElementById('pubsubControls').style.display = 'block';
                console.log(`Connected to PubSub channel: ${channel}`);
            };
            wsPubSub.onmessage = function(event) {
                const output = document.getElementById('pubsubOutput');
                output.innerHTML += `<p>${event.data}</p>`;
            };
            wsPubSub.onclose = function() {
                console.log('PubSub WebSocket closed.');
                document.getElementById('pubsubControls').style.display = 'none';
            };
        });

        document.getElementById('sendPubSub').addEventListener('click', function() {
            const message = document.getElementById('pubsubMessage').value;
            if (wsPubSub && wsPubSub.readyState === WebSocket.OPEN) {
                wsPubSub.send(message);
            } else {
                alert("WebSocket is not connected.");
            }
        });

        document.getElementById('connectCommand').addEventListener('click', function() {
            wsCommand = new WebSocket(`ws://localhost:8000/ws`);
            wsCommand.onopen = function() {
                document.getElementById('commandControls').style.display = 'block';
                console.log('Connected to Command WebSocket.');
            };
            wsCommand.onmessage = function(event) {
                const output = document.getElementById('generalOutput');
                output.innerHTML += `<p>${event.data}</p>`;
            };
            wsCommand.onclose = function() {
                console.log('Command WebSocket closed.');
                document.getElementById('commandControls').style.display = 'none';
            };
        });

        document.getElementById('sendCommand').addEventListener('click', function() {
            const command = document.getElementById('command').value;
            const key = document.getElementById('key').value;
            const value = document.getElementById('value').value;

            // Construct the message in JSON format based on the selected command
            let message = {};
            if (command === 'SET') {
                if (!key || !value) {
                    alert("SET command requires both key and value.");
                    return;
                }
                message = { command, key, value };
            } else if (command === 'GET' || command === 'DEL') {
                if (!key) {
                    alert(`${command} command requires a key.`);
                    return;
                }
                message = { command, key };
            } else if (command === 'KEYS') {
                message = { command, key: key || '*' };  // Default to '*' if no key is provided
            }

            // Send the JSON message if the WebSocket is open
            if (wsCommand && wsCommand.readyState === WebSocket.OPEN) {
                wsCommand.send(JSON.stringify(message));
            } else {
                alert("WebSocket is not connected.");
            }
        });
    </script>
</body>
</html>

    """
    return html_content

@app.websocket("/ws/pubsub/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    await websocket.accept()
    pubsub = redis_client.pubsub()
    shared_data["count"] += 1
    await pubsub.subscribe(channel)  # Subscribe to the Redis channel using the parameter
    
    async def send_messages():
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"])
    
    # Create a background task for sending messages
    send_task = asyncio.create_task(send_messages())
    
    try:
        while True:
            data = await websocket.receive_text()
            await redis_client.publish(channel, data)  # Publish messages to Redis using the parameter
    except Exception as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        shared_data["count"] -= 1
        send_task.cancel()  # Clean up the background task when the connection is closed
        await pubsub.unsubscribe(channel)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    shared_data["count"] += 1

    try:
        while True:
            # Receive data from the WebSocket client
            data = await websocket.receive_text()
            
            # Parse incoming data (expecting JSON-formatted string)
            try:
                command_data:dict = json.loads(data)
                command = command_data.get("command")
                key = command_data.get("key")
                value = command_data.get("value")
            except json.JSONDecodeError:
                await websocket.send_text("Invalid input. Expected JSON formatted data.")
                continue

            if command == "SET" and key and value:
                # Set key-value pair in Redis
                await redis_client.set(key, value)
                await websocket.send_text(f"SET: {key} = {value}")

            elif command == "GET" and key:
                # Get value for a key from Redis
                result = await redis_client.get(key)
                if result is None:
                    await websocket.send_text(f"GET: {key} does not exist")
                else:
                    await websocket.send_text(f"GET: {key} = {result}")

            elif command == "DEL" and key:
                # Delete a key from Redis
                deleted_count = await redis_client.delete(key)
                if deleted_count > 0:
                    await websocket.send_text(f"DEL: {key} deleted")
                else:
                    await websocket.send_text(f"DEL: {key} does not exist")

            elif command == "KEYS":
                # Get all keys matching a pattern (default pattern is '*')
                pattern = key if key else '*'
                keys = await redis_client.keys(pattern)
                await websocket.send_text(f"KEYS: {keys}")

            else:
                # Unsupported command or missing parameters
                await websocket.send_text("Unsupported command or missing parameters. Use SET, GET, DEL, KEYS.")
    
    except Exception as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        shared_data["count"] -= 1
