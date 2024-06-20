
import json
import gradio as gr
import aiohttp
import asyncio

# FastAPI server base URL
FastAPI_URL = "http://127.0.0.1:8000"
Redis_URL = "redis://127.0.0.1:6379"

async def send_get_request(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
        
def set_urls(fastapi=FastAPI_URL,redis=Redis_URL):
    global FastAPI_URL,Redis_URL
    FastAPI_URL,Redis_URL = fastapi,redis
    return json.dumps(dict(FastAPI_URL=FastAPI_URL,Redis_URL=Redis_URL))

def task_status(task_id):
    return asyncio.run(send_get_request(f"{FastAPI_URL}/task_status/{task_id}"))

def stop_task(task_id):
    return asyncio.run(send_get_request(f"{FastAPI_URL}/stop_task/{task_id}"))

def start_video_stream(video_src, fps, width, height, redis_stream_key):
    url = f"{FastAPI_URL}/start_video_stream/?video_src={video_src}&fps={fps}&width={width}&height={height}&redis_stream_key={redis_stream_key}&redis_url={Redis_URL}"
    return asyncio.run(send_get_request(url))

def stop_video_stream(redis_stream_key, ):
    url = f"{FastAPI_URL}/stop_video_stream/?redis_stream_key={redis_stream_key}&redis_url={Redis_URL}"
    return asyncio.run(send_get_request(url))

def stop_all_stream():
    url = f"{FastAPI_URL}/stop_all_stream/?redis_url={Redis_URL}"
    return asyncio.run(send_get_request(url))

def video_stream_info():
    return asyncio.run(send_get_request(f"{FastAPI_URL}/video_stream_info/?redis_url={Redis_URL}"))

def clone_stream(read_stream_key, write_stream_key):
    url = f"{FastAPI_URL}/clone_stream/?redis_url={Redis_URL}&read_stream_key={read_stream_key}&write_stream_key={write_stream_key}"
    return asyncio.run(send_get_request(url))

def cv_resize_stream(w, h, read_stream_key, write_stream_key):
    url = f"{FastAPI_URL}/cv_resize_stream/?w={w}&h={h}&redis_url={Redis_URL}&read_stream_key={read_stream_key}&write_stream_key={write_stream_key}"
    return asyncio.run(send_get_request(url))

def yolo_image_stream(read_stream_key, write_stream_key, modelname, conf):
    url = f"{FastAPI_URL}/yolo_image_stream/?redis_url={Redis_URL}&read_stream_key={read_stream_key}&write_stream_key={write_stream_key}&modelname={modelname}&conf={conf}"
    return asyncio.run(send_get_request(url))

def web_image_show(stream_key):
    # Replace localhost and port if your server is running on a different URL
    url = f"{FastAPI_URL}/web_image_show/{stream_key}"
    html_content = f"""
    <html>
    <body>
    <!-- Stream the image data into an img tag -->
    <img src="{url}" style="width: 100%;">
    </body>
    </html>
    """
    return html_content

# Creating Gradio interfaces for each function
iface_set_urls = gr.Interface(fn=set_urls, inputs=["text", "text"], examples=[[FastAPI_URL,Redis_URL]],
                                outputs="json", description="Set API URLs.")

iface_task_status = gr.Interface(fn=task_status, inputs="text",outputs="json", description="Check the status of a video capture task.")

iface_stop_task = gr.Interface(fn=stop_task, inputs="text",outputs="json", description="Terminate a running task.")

iface_start_video_stream = gr.Interface(fn=start_video_stream, inputs=["text", "number", "number", "number", "text"],
                                examples=[["d:/Download/Driving from Hell's Kitchen Manhattan to Newark Liberty International Airport.mp4",
                                           60,1920,1080,"camera:0"]], 
                                outputs="json", description="Start a video stream task.")

iface_stop_video_stream = gr.Interface(fn=stop_video_stream, inputs=["text"],
                                examples=[["camera:0"]], 
                                outputs="json", description="Stop a video stream task.")

iface_stop_all_stream = gr.Interface(fn=stop_all_stream, inputs=[], outputs="json", description="Stop all video stream tasks.")

iface_video_stream_info = gr.Interface(fn=video_stream_info, inputs=[],outputs="json", description="Get video stream information.")

iface_clone_stream = gr.Interface(fn=clone_stream, inputs=["text", "text"],
                                examples=[["camera:0","clone:0"]], 
                                outputs="json", description="Start a clone stream task.")

iface_cv_resize_stream = gr.Interface(fn=cv_resize_stream, inputs=["number", "number", "text", "text"],
                                examples=[[192,108,"camera:0","resize:0"]], 
                                outputs="json", description="Start a resize stream task.")

iface_yolo_image_stream = gr.Interface(fn=yolo_image_stream, inputs=["text", "text", "text", "number"],
                                examples=[["camera:0","ai:0","yolov5s6",0.6]], 
                                outputs="json", description="Start a YOLO stream task.")

iface_web_image_show = gr.Interface(fn=web_image_show,inputs=gr.Textbox(label="Enter Stream Key"),outputs=gr.HTML(label="Live Video Stream"),
                                examples=[["camera:0"]], 
                                title="Live Video Stream Viewer",description="Enter the stream key to view the live video stream.")

# Combine all interfaces into tabs
tabbed_interface = gr.TabbedInterface([iface_set_urls, iface_task_status, iface_stop_task, iface_start_video_stream, iface_stop_video_stream, iface_stop_all_stream, iface_video_stream_info, iface_clone_stream, iface_cv_resize_stream, iface_yolo_image_stream,iface_web_image_show], 
                                      ["Set URLs", "Task Status", "Stop Task", "Start Video Stream", "Stop Video Stream", "Stop All Streams", "Video Stream Info", "Clone Stream", "Resize Stream", "YOLO Image Stream", "Web Image Show"])
tabbed_interface.launch(server_name='0.0.0.0')
