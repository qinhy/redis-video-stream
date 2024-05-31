import json
from fastapi import FastAPI, HTTPException
from celery.result import AsyncResult
from celery_task import CeleryTaskManager,celery_app
from typing import Optional
import redis
from urllib.parse import urlparse


app = FastAPI()

@app.get("/task_status/{task_id}")
async def task_status(task_id: str):
    """
    Check the status of a video capture task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    result = {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result
    }
    return result

@app.post("/stop_task/{task_id}")
async def stop_task(task_id: str):
    """
    Attempt to terminate a running task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    if not task_result.ready():
        task_result.revoke(terminate=True)
        return {"message": "Task termination requested", "task_id": task_id}
    else:
        return {"message": "Task already completed or does not exist", "task_id": task_id}

@app.post("/start_video_stream/")
async def start_video_stream(video_src: str = '0', fps: float = 30.0, width: int =800, height: int =600, fmt: str = '.jpg',
                             redis_stream_key: str = 'camera:0', redis_url: str = 'redis://127.0.0.1:6379',  maxlen: int = 10):
    """
    Start a video stream task.
    """
    task = CeleryTaskManager.start_video_stream.delay(video_src, fps, width, height, fmt,
                                                      redis_stream_key, redis_url, maxlen)
    return {"message": "Task started", "task_id": task.id}

@app.post("/stop_video_stream/")
async def stop_video_stream(redis_stream_key: str = 'camera:0', redis_url: str = 'redis://127.0.0.1:6379'):
    """
    stop a video stream task.
    """
    task = CeleryTaskManager.stop_video_stream.delay(redis_stream_key, redis_url)
    return {"message": "Task stop", "task_id": task.id}

@app.post("/video_stream_info/")
async def video_stream_info(redis_url: str = 'redis://127.0.0.1:6379'):
    url = urlparse(redis_url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    info = {}
    for k in conn.keys(f'info:*'):
        info[k] = json.loads(conn.get(k))
    return info

@app.post("/cvshow_image_stream/")
async def cvshow_image_stream(redis_url: str = 'redis://127.0.0.1:6379', stream_key: str = 'camera:0', batch_size: int = 10):
    """
    Start reading image stream from Redis.
    """
    try:
        task = CeleryTaskManager.cvshow_image_stream.delay(redis_url, stream_key, batch_size)
        return {"message": "Stream cvshow task started", "task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))