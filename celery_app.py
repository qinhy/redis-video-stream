from celery import Celery

app = Celery('video_tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

app.conf.task_routes = {'video_capture.capture_video': {'queue': 'video'}}
