import time
from celery_task import CeleryTaskManager,get_video_stream_info

redis_url='redis://127.0.0.1:6379'

task = CeleryTaskManager.start_video_stream.delay(
    # video_src="d:/Download/Driving from Hell's Kitchen Manhattan to Newark Liberty International Airport.mp4",
    video_src="chrome",
    fps=30, width=1920, height=1080,
    redis_stream_key='camera:0', redis_url=redis_url)

while True:
    info = get_video_stream_info()
    if 'info:camera:0' in info:
        break
    time.sleep(1)
print({"message": "video stream started", "task_id": task.id})

task = CeleryTaskManager.yolo_image_stream.delay(redis_url=redis_url,read_stream_key='camera:0',
                                    write_stream_key='ai:0',modelname='yolov5s6',conf=0.6)

while True:
    info = get_video_stream_info()
    if 'info:ai:0' in info:
        break
    time.sleep(1)
print({"message": "yolo stream task started", "task_id": task.id})

task = CeleryTaskManager.cv_resize_stream.delay(redis_url=redis_url,read_stream_key='ai:0',
                                                write_stream_key='resize:0',w=192,h=108)
print({"message": "resize stream task started", "task_id": task.id})
