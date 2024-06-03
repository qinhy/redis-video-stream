import json
from fastapi import FastAPI, HTTPException
from celery.result import AsyncResult
from celery_task import CeleryTaskManager,celery_app
from typing import Optional
import redis
from urllib.parse import urlparse

CeleryTaskManager.stop_all_stream.delay('redis://127.0.0.1:6379')

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

@app.get("/stop_task/{task_id}")
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

@app.get("/start_video_stream/")
async def start_video_stream(video_src: str = '0', fps: float = 30.0, width: int =848, height: int =480,
                             redis_stream_key: str = 'camera:0', redis_url: str = 'redis://127.0.0.1:6379'):
    """
    Start a video stream task.
    """
    task = CeleryTaskManager.start_video_stream.delay(video_src, fps, width, height,
                                                      redis_stream_key, redis_url)
    return {"message": "video stream started", "task_id": task.id}

@app.get("/stop_video_stream/")
async def stop_video_stream(redis_stream_key: str = 'camera:0', redis_url: str = 'redis://127.0.0.1:6379'):
    """
    stop a video stream task.
    """
    task = CeleryTaskManager.stop_video_stream.delay(redis_stream_key, redis_url)
    return {"message": "stream stop", "task_id": task.id}

@app.get("/stop_all_stream/")
async def stop_all_stream(redis_url: str = 'redis://127.0.0.1:6379'):
    """
    stop all video stream task.
    """
    task = CeleryTaskManager.stop_all_stream.delay(redis_url)
    return {"message": "all stream stop", "task_id": task.id}

@app.get("/video_stream_info/")
async def video_stream_info(redis_url: str = 'redis://127.0.0.1:6379'):
    url = urlparse(redis_url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    info = {}
    for k in conn.keys(f'info:*'):
        info[k] = json.loads(conn.get(k))
    return info

@app.get("/clone_stream/")
async def clone_stream(redis_url: str='redis://127.0.0.1:6379', read_stream_key: str='camera:0',
                               write_stream_key: str='clone:0'):
    """
    Start a clone stream task.
    """
    task = CeleryTaskManager.clone_stream.delay(redis_url=redis_url,read_stream_key=read_stream_key,
                                                write_stream_key=write_stream_key)
    return {"message": "clone stream started", "task_id": task.id}

@app.get("/flip_stream/")
async def flip_stream(redis_url: str='redis://127.0.0.1:6379', read_stream_key: str='camera:0',
                               write_stream_key: str='flip:0'):
    """
    Start a flip stream task.
    """
    task = CeleryTaskManager.flip_stream.delay(redis_url=redis_url,read_stream_key=read_stream_key,
                                                write_stream_key=write_stream_key)
    return {"message": "flip stream started", "task_id": task.id}

@app.get("/split_stream/")
async def split_stream(a:int,b:int,c:int,d:int,redis_url: str='redis://127.0.0.1:6379', read_stream_key: str='camera:0',
                               write_stream_key: str='split:0'):
    """
    Start a split stream task.
    """
    task = CeleryTaskManager.split_stream.delay(bbox=[a,b,c,d],redis_url=redis_url,read_stream_key=read_stream_key,
                                                write_stream_key=write_stream_key)
    return {"message": "split stream started", "task_id": task.id}

@app.get("/cv_resize_stream/")
async def cv_resize_stream(w:int=100,h:int=100, redis_url: str='redis://127.0.0.1:6379', read_stream_key: str='camera:0',
                               write_stream_key: str='resize:0'):
    """
    Start a resize stream task.
    """
    task = CeleryTaskManager.cv_resize_stream.delay(redis_url=redis_url,read_stream_key=read_stream_key,
                                                         write_stream_key=write_stream_key,w=w,h=h)
    return {"message": "resize stream started", "task_id": task.id}


@app.get("/yolo_image_stream/")
async def yolo_image_stream(redis_url: str='redis://127.0.0.1:6379', read_stream_key: str='camera:0',
                                            write_stream_key: str='ai:0',modelname:str='yolov5s6',conf:float=0.6,
                                        ):
    """
    Start a yolo stream task.
    """
    task = CeleryTaskManager.yolo_image_stream.delay(redis_url=redis_url,read_stream_key=read_stream_key,
                                                         write_stream_key=write_stream_key,modelname=modelname,conf=conf)
    return {"message": "yolo stream started", "task_id": task.id}

@app.get("/cvshow_image_stream/")
async def cvshow_image_stream(redis_url: str = 'redis://127.0.0.1:6379', stream_key: str = 'camera:0'):
    """
    Start reading image stream from Redis.
    """
    try:
        task = CeleryTaskManager.cvshow_image_stream.delay(redis_url, stream_key)
        return {"message": "Stream cvshow task started", "task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))