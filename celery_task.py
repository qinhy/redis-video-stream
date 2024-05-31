import base64
import json
import os
import random
import subprocess
import sys
from pathlib import Path
import time
from typing import List
import numpy as np
import cv2
import time
import redis
from urllib.parse import urlparse

from celery import Celery
from celery.app import task as Task

####################################################################################
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')
IP = '127.0.0.1'
os.environ.setdefault('CELERY_BROKER_URL', 'redis://'+IP)
os.environ.setdefault('CELERY_RESULT_BACKEND', 'redis://'+IP+'/0')
# os.environ.setdefault('CELERY_BROKER_URL', f'amqp://guest@{IP}//')
# os.environ.setdefault('CELERY_RESULT_BACKEND', f'amqp://guest@{IP}//')
# os.environ.setdefault('CELERY_TASK_SERIALIZER', 'json')
####################################################################################
celery_app = Celery('tasks')  # , broker='redis://'+IP)

def WrappTask(task:Task):
    def update_progress_state(progress=1.0,msg=''):
        task.update_state(state='PROGRESS',meta={'progress': progress,'msg':msg})
        task.send_event('task-progress', result={'progress': progress})
        
    def update_error_state(error='null'):
        task.update_state(state='FAILURE',meta={'error': error})
    
    task.progress = update_progress_state
    task.error = update_error_state
    return task 

class VideoGenerator:
    def __init__(self, video_src=0, fps=30.0, width=800, height=600):
        self.video_src = video_src if video_src is not None else 0
        self.isFile = not str(self.video_src).isdecimal()
        if not self.isFile:
            self.video_src = int(self.video_src)
        self.cam = cv2.VideoCapture(self.video_src)
        if not self.isFile:
            self.cam.set(cv2.CAP_PROP_FPS, fps)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            self.fps = self.cam.get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        return self

    def __next__(self):
        ret_val, img = self.cam.read()
        if not ret_val:
            raise StopIteration()
        return cv2.flip(img, 1) if not self.isFile else img

class RedisStreamWriter:
    def __init__(self, redis_url, stream_key, maxlen=1000, fmt='.jpg'):
        url = urlparse(redis_url)
        self.conn = redis.Redis(host=url.hostname, port=url.port)
        self.stream_key = stream_key
        self.maxlen = maxlen
        self.fmt = fmt

    def write_to_stream(self, image, count):
        _, buffer = cv2.imencode(self.fmt, image)
        message = {'image': buffer.tobytes(), 'count': count}
        self.conn.xadd(self.stream_key, message, maxlen=self.maxlen)

class RedisStreamReader:
    def __init__(self, redis_url, stream_key):
        url = urlparse(redis_url)
        self.conn = redis.Redis(host=url.hostname, port=url.port)
        self.stream_key = stream_key

    def read_raw_from_stream(self,count=10):
        last_id = '0-0'
        while True:
            messages = self.conn.xread({self.stream_key: last_id},count=count, block=1000)
            if messages:
                for _, records in messages:
                    for record in records:
                        # yield record[1]['image'], record[0]  # return image data and message id
                        message_id, data = record
                        last_id = message_id  # Update last_id to the latest message_id read
                        yield record

    def read_image_from_stream(self,count=10):
        for record in self.read_raw_from_stream(count=count):
            message_id, data = record
            image_bytes = data[b'image']  # Assuming the image is stored under key 'image'
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            yield image

class CeleryTaskManager:    
    @staticmethod
    @celery_app.task(bind=True)
    def revoke(t: Task, task_id: str):
        """Method to revoke a task."""
        return celery_app.control.revoke(task_id, terminate=True)
    
    @staticmethod
    @celery_app.task(bind=True)    
    def start_video_stream(t: Task, video_src:str, fps:float, width=800, height=600, fmt='.jpg',
                                    redis_stream_key='camera-stream:0', redis_url='redis://127.0.0.1:6379', maxlen=10):
        url = urlparse(redis_url)
        conn = redis.Redis(host=url.hostname, port=url.port)
        if conn.exists(redis_stream_key):
            info = conn.get(f'info:{redis_stream_key}')
            info = json.loads(info)
            return f"Stream {redis_stream_key} is already running. By task id of {info['task_id']}"

        conn.set(f'info:{redis_stream_key}',json.dumps(dict(task_id=t.request.id,
                                                            video_src=video_src, fps=fps, width=width, height=height, fmt=fmt)))
        conn.close()
        # Initialize video generator and writer
        generator = VideoGenerator(video_src=video_src, fps=fps, width=width, height=height)
        writer = RedisStreamWriter(redis_url, redis_stream_key, maxlen=maxlen, fmt=fmt)
        print(f"Stream {redis_stream_key} started. By task id of {t.request.id}")
        for image in generator:
            writer.write_to_stream(image, generator.cam.get(cv2.CAP_PROP_POS_FRAMES))     

    @staticmethod
    @celery_app.task(bind=True)    
    def stop_video_stream(t: Task, redis_stream_key='camera-stream:0', redis_url='redis://127.0.0.1:6379'):
        url = urlparse(redis_url)
        conn = redis.Redis(host=url.hostname, port=url.port)
        info = conn.get(f'info:{redis_stream_key}')
        if info is None:
            return {'msg':'no such stream'}
        info = json.loads(info)
        celery_app.control.revoke(info['task_id'], terminate=True)
        conn.delete(redis_stream_key)
        conn.delete(f'info:{redis_stream_key}')
        return {'msg':f'delete stream {redis_stream_key}'}
        
    @staticmethod
    @celery_app.task(bind=True)
    def cvshow_image_stream(t: Task, redis_url:str, stream_key:str, batch_size=10):
        reader = RedisStreamReader(redis_url=redis_url,stream_key=stream_key)
        frame_count = 0
        start_time = time.time()
        for image in reader.read_image_from_stream(batch_size):
            # Process the image as needed
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            frame_count += 1

            # Display FPS on the image
            cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Streamed Image', image)
            cv2.waitKey(1)  # Display the image for a brief moment