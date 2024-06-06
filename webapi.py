import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from celery.result import AsyncResult
import redis
from celery_task import CeleryTaskManager, RedisStreamReader,celery_app, get_video_stream_info

CeleryTaskManager.stop_all_stream.delay('redis://127.0.0.1:6379')
redis.Redis(host='127.0.0.1', port=6379).flushdb()

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
    return {"message": "video stream task started", "task_id": task.id}

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
    return get_video_stream_info(redis_url)

@app.get("/clone_stream/")
async def clone_stream(redis_url: str='redis://127.0.0.1:6379', read_stream_key: str='camera:0',
                               write_stream_key: str='clone:0'):
    """
    Start a clone stream task.
    """
    task = CeleryTaskManager.clone_stream.delay(redis_url=redis_url,read_stream_key=read_stream_key,
                                                write_stream_key=write_stream_key)
    return {"message": "clone stream task started", "task_id": task.id}

@app.get("/flip_stream/")
async def flip_stream(redis_url: str='redis://127.0.0.1:6379', read_stream_key: str='camera:0',
                               write_stream_key: str='flip:0'):
    """
    Start a flip stream task.
    """
    task = CeleryTaskManager.flip_stream.delay(redis_url=redis_url,read_stream_key=read_stream_key,
                                                write_stream_key=write_stream_key)
    return {"message": "flip stream task started", "task_id": task.id}

@app.get("/split_stream/")
async def split_stream(a:int,b:int,c:int,d:int,redis_url: str='redis://127.0.0.1:6379', read_stream_key: str='camera:0',
                               write_stream_key: str='split:0'):
    """
    Start a split stream task.
    """
    task = CeleryTaskManager.split_stream.delay(bbox=[a,b,c,d],redis_url=redis_url,read_stream_key=read_stream_key,
                                                write_stream_key=write_stream_key)
    return {"message": "split stream task started", "task_id": task.id}

@app.get("/cv_resize_stream/")
async def cv_resize_stream(w:int=100,h:int=100, redis_url: str='redis://127.0.0.1:6379', read_stream_key: str='camera:0',
                               write_stream_key: str='resize:0'):
    """
    Start a resize stream task.
    """
    task = CeleryTaskManager.cv_resize_stream.delay(redis_url=redis_url,read_stream_key=read_stream_key,
                                                         write_stream_key=write_stream_key,w=w,h=h)
    return {"message": "resize stream task started", "task_id": task.id}


@app.get("/yolo_image_stream/")
async def yolo_image_stream(redis_url: str='redis://127.0.0.1:6379', read_stream_key: str='camera:0',
                                            write_stream_key: str='ai:0',modelname:str='yolov5s6',conf:float=0.6,
                                        ):
    """
    Start a yolo stream task.
    """
    task = CeleryTaskManager.yolo_image_stream.delay(redis_url=redis_url,read_stream_key=read_stream_key,
                                                         write_stream_key=write_stream_key,modelname=modelname,conf=conf)
    return {"message": "yolo stream task started", "task_id": task.id}

@app.get("/cvshow_image_stream/")
async def cvshow_image_stream(redis_url: str = 'redis://127.0.0.1:6379', stream_key: str = 'camera:0', fontscale:float=1):
    """
    Start reading image stream from Redis.
    """
    try:
        task = CeleryTaskManager.cvshow_image_stream.delay(redis_url, stream_key, fontscale)
        return {"message": "Stream cvshow task started", "task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/web_image_show/{read_stream_key}")
def web_image_show(read_stream_key:str='camera:0',redis_url: str = 'redis://127.0.0.1:6379'):

    reader = RedisStreamReader(redis_url=redis_url, stream_key=read_stream_key)
    def generate_frames():
        for frame_count,(image,metadata) in enumerate(reader.read_stream_generator()):
            # Convert the Numpy array to a format that can be sent over the network
            _, buffer = cv2.imencode('.png', image)
            frame = buffer.tobytes()            
            # Use multipart/x-mixed-replace with boundary frame to keep the connection open
            yield (b'--frame\r\n'
                b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

    # Create a StreamingResponse that sends the image to the client
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")
