import asyncio
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

redis_client = redis.Redis(host='localhost', decode_responses=False)  # Use binary data

@app.get("/")
async def get_homepage():
    return HTMLResponse(r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebSocket Image Client</title>
</head>
<body>
    <h1>WebSocket Image Client</h1>
    <div>
        <label for="width">Width:</label>
        <input type="number" id="width" value="1920" min="1">
        <label for="height">Height:</label>
        <input type="number" id="height" value="1080" min="1">
    </div>
    <button id="sendImageBtn">Send Random Image</button>
    <p id="status"></p>
    <div>
        <h3>Sent Image:</h3>
        <canvas id="sentImageCanvas" style="border:1px solid #000;"></canvas>
    </div>
    <div>
        <h3>Received Image:</h3>
        <img id="receivedImage" alt="No image received yet" border:1px solid #000;">
    </div>

    <script>
        function generateRandomImage(width, height) {
            // Create a canvas element
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const context = canvas.getContext('2d');

            // Generate random pixels
            const imageData = context.createImageData(width, height);
            for (let i = 0; i < imageData.data.length; i += 4) {
                imageData.data[i] = Math.floor(Math.random() * 256);     // Red
                imageData.data[i + 1] = Math.floor(Math.random() * 256); // Green
                imageData.data[i + 2] = Math.floor(Math.random() * 256); // Blue
                imageData.data[i + 3] = 255;                             // Alpha (fully opaque)
            }
            context.putImageData(imageData, 0, 0);

            // Display the generated image in the "Sent Image" canvas
            const sentImageCanvas = document.getElementById('sentImageCanvas');
            sentImageCanvas.width = width;
            sentImageCanvas.height = height;
            const sentContext = sentImageCanvas.getContext('2d');
            sentContext.putImageData(imageData, 0, 0);

            // Convert canvas to a blob (binary data)
            return new Promise((resolve) => {
                canvas.toBlob((blob) => {
                    resolve(blob);
                }, 'image/jpeg');
            });
        }

        async function connectWebSocket() {
            const statusElement = document.getElementById('status');
            const receivedImageElement = document.getElementById('receivedImage');
            const websocket = new WebSocket('ws://localhost:8000/ws');

            websocket.onopen = () => {
                statusElement.textContent = 'Connected to WebSocket server';
            };

            websocket.onmessage = async (event) => {
                // Handle receiving binary data and display it as an image
                const blob = event.data instanceof Blob ? event.data : new Blob([event.data]);
                const objectURL = URL.createObjectURL(blob);
                receivedImageElement.src = objectURL;
                statusElement.textContent = `Received image of size: ${blob.size || event.data.length} bytes`;
            };

            websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                statusElement.textContent = 'WebSocket error occurred';
            };

            websocket.onclose = () => {
                statusElement.textContent = 'WebSocket connection closed';
            };

            document.getElementById('sendImageBtn').addEventListener('click', async () => {
                const width = parseInt(document.getElementById('width').value, 10);
                const height = parseInt(document.getElementById('height').value, 10);

                if (isNaN(width) || isNaN(height) || width <= 0 || height <= 0) {
                    statusElement.textContent = 'Please enter valid dimensions for the image';
                    return;
                }

                const blob = await generateRandomImage(width, height); // Generate a random image with specified size
                const arrayBuffer = await blob.arrayBuffer(); // Convert Blob to ArrayBuffer
                websocket.send(arrayBuffer); // Send image data as binary
                statusElement.textContent = `Sent image of size: ${blob.size} bytes`;
            });
        }

        // Connect to the WebSocket when the page loads
        window.onload = connectWebSocket;
    </script>
</body>
</html>


""")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pubsub = redis_client.pubsub()

    async def listen_to_channel():
        await pubsub.subscribe("video_channel")
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_bytes(message["data"])  # Send received data to the client

    listen_task = asyncio.create_task(listen_to_channel())

    try:
        while True:
            # Receive video frame as bytes from the client
            data = await websocket.receive_bytes()
            # Publish to Redis channel
            await redis_client.publish("video_channel", data)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        listen_task.cancel()
        # await websocket.close()
