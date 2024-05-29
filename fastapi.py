from fastapi import FastAPI, HTTPException
from celery.result import AsyncResult
from celery_app import app as celery_app
from typing import Optional

app = FastAPI()

@app.post("/start_capture/")
async def start_capture(infile: Optional[int] = None, fps: float = 30.0, output: str = 'camera:0', url: str = 'redis://127.0.0.1:6379', fmt: str = '.jpg', maxlen: int = 10000, count: Optional[int] = None):
    """
    Start a video capture task.
    """
    task = celery_app.send_task('video_capture.capture_video', args=[infile, fps, output, url, fmt, maxlen, count])
    return {"message": "Task started", "task_id": task.id}

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

@app.post("/stop_capture/{task_id}")
async def stop_capture(task_id: str):
    """
    Attempt to terminate a running video capture task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    if not task_result.ready():
        task_result.revoke(terminate=True)
        return {"message": "Task termination requested", "task_id": task_id}
    else:
        return {"message": "Task already completed or does not exist", "task_id": task_id}
