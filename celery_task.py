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
import time
import cv2
import torch
import json
import numpy as np
import cv2
from multiprocessing import Lock, shared_memory

from celery import Celery
from celery.app import task as Task

####################################################################################
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')
IP = '127.0.0.1'
os.environ.setdefault('CELERY_BROKER_URL', 'redis://'+IP)
os.environ.setdefault('CELERY_RESULT_BACKEND', 'redis://'+IP+'/0')
####################################################################################
celery_app = Celery('tasks')  

def getredis(redis_url):
    url = urlparse(redis_url)
    return redis.Redis(host=url.hostname, port=url.port)

def is_stream_exists(conn,stream_key):
        if conn.exists(stream_key):
            info = conn.get(f'info:{stream_key}')
            if info:
                info = json.loads(info)
                return f"Stream {stream_key} is already running. By task id of {info['task_id']}"
        return False

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
    
class SharedMemoryStreamWriter:
    def __init__(self, shm_name, array_shape, dtype=np.uint8):
        self.array_shape = array_shape
        self.dtype = dtype
        # Calculate the buffer size needed for the array
        self.shm_size = int(np.prod(array_shape) * np.dtype(dtype).itemsize)
        # Create the shared memory
        self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.shm_size)
        # Create the numpy array with the buffer from shared memory
        self.buffer = np.ndarray(array_shape, dtype=dtype, buffer=self.shm.buf)

    def write_to_stream(self, image):
        # Ensure the image fits the predefined array shape and dtype
        if image.shape != self.array_shape:
            raise ValueError(f"Image does not match the predefined shape.({image.shape}!={self.array_shape})")
        if image.dtype != self.dtype:
            raise ValueError(f"Image does not match the predefined data type.({image.dtype}!={self.dtype})")
        # Write the image to the shared memory buffer
        np.copyto(self.buffer, image)

    def close(self):
        self.shm.close()
        self.shm.unlink()

class SharedMemoryStreamReader:
    def __init__(self, shm_name, array_shape, dtype=np.uint8):
        self.array_shape = array_shape
        self.dtype = dtype
        # Attach to the existing shared memory
        self.shm = shared_memory.SharedMemory(name=shm_name)
        # Map the numpy array to the shared memory
        self.buffer = np.ndarray(array_shape, dtype=dtype, buffer=self.shm.buf)

    def read_from_stream(self):
        # Simply return a copy of the array to ensure the data remains consistent
        return np.copy(self.buffer)

    def close(self):
        self.shm.close()


class RedisStreamWriter:
    def __init__(self, redis_url, stream_key, shape, maxlen=1000, fmt='.jpg'):
        self.conn = getredis(redis_url)
        self.stream_key = stream_key
        self.maxlen = maxlen
        self.fmt = fmt
        self.smwriter = SharedMemoryStreamWriter(stream_key, shape)  # 10MB shared memory size

    def write_to_stream(self, image, metadata={}):
        # _, buffer = cv2.imencode(self.fmt, image)
        # message = {'image': buffer.tobytes(), 'metadata': json.dumps(metadata)}
        message = {'image': '', 'metadata': json.dumps(metadata)}
        
        self.smwriter.write_to_stream(image)
        self.conn.xadd(self.stream_key, message, maxlen=self.maxlen)

    def close(self):
        self.smwriter.close()

class RedisStreamReader:
    def __init__(self, redis_url, stream_key):
        self.conn = getredis(redis_url)
        self.stream_key = stream_key
        self.smreader = None#SharedMemoryStreamReader(stream_key, (720,1280,3))  # Same size as used by writer

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
            # image_bytes = data[b'image']  # Assuming the image is stored under key 'image'
            # image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            # image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if self.smreader is None:
                metadata = json.loads(data[b'metadata'])
                self.smreader = SharedMemoryStreamReader(self.stream_key, metadata['shape'])  # Same size as used by writer
            image = self.smreader.read_from_stream()
            yield image
    
    def close(self):
        self.smreader.close()

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
        conn = getredis(redis_url)
        if is_stream_exists(conn,redis_stream_key):return is_stream_exists(conn,redis_stream_key)

        conn.set(f'info:{redis_stream_key}',json.dumps(dict(task_id=t.request.id,
                                                            video_src=video_src, fps=fps, width=width, height=height, fmt=fmt)))
        conn.close()
        # Initialize video generator and writer
        generator = VideoGenerator(video_src=video_src, fps=fps, width=width, height=height)
        writer = RedisStreamWriter(redis_url, redis_stream_key, (height,width,3), maxlen=maxlen, fmt=fmt)
        print(f"Stream {redis_stream_key} started. By task id of {t.request.id}")

        frame_count = 0
        start_time = time.time()
        for image in generator:
            writer.write_to_stream(image, dict(count=generator.cam.get(cv2.CAP_PROP_POS_FRAMES),shape=image.shape))     

            # Process the image as needed
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            frame_count += 1

            # Display FPS on the image
            cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(f'Streamed Image Raw {redis_stream_key}', image)
            if cv2.waitKey(1) == 27:  # ESC key
                cv2.destroyAllWindows()
                break
        writer.close()


    @staticmethod
    @celery_app.task(bind=True)    
    def stop_video_stream(t: Task, redis_stream_key='camera-stream:0', redis_url='redis://127.0.0.1:6379'):
        conn = getredis(redis_url)
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
    def stop_all_stream(t: Task, redis_url='redis://127.0.0.1:6379'):
        url = urlparse(redis_url)
        conn = redis.Redis(host=url.hostname, port=url.port)
        info = {}
        for k in conn.keys(f'info:*'):
            redis_stream_key = k.decode().replace('info:','')
            info = conn.get(f'info:{redis_stream_key}')
            info = json.loads(info)
            celery_app.control.revoke(info['task_id'], terminate=True)
            conn.delete(redis_stream_key)
            conn.delete(f'info:{redis_stream_key}')
            return {'msg':f'delete stream {redis_stream_key}'}
       
    @staticmethod
    @celery_app.task(bind=True)
    def cvshow_image_stream(t: Task, redis_url:str, stream_key:str):
        reader = RedisStreamReader(redis_url=redis_url,stream_key=stream_key)
        frame_count = 0
        start_time = time.time()
        for image in reader.read_image_from_stream():
            # Process the image as needed
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            frame_count += 1

            # Display FPS on the image
            cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(f'Streamed Image {stream_key}', image)
            if cv2.waitKey(1) == 27:  # ESC key
                cv2.destroyAllWindows()
                break
        reader.close()
            
    @staticmethod
    def stream2stream(image_processor=lambda i,image:(image,dict(count=i)),
                      redis_url: str='redis://127.0.0.1:6379',read_stream_key: str='camera-stream:0',
                      write_stream_key: str='out-stream:0',maxlen:int=10,fmt: str='.jpg',
                      metadaata={}):
        
        conn = getredis(redis_url)
        if is_stream_exists(conn,write_stream_key):return is_stream_exists(conn,write_stream_key)        
        conn.set(f'info:{write_stream_key}',json.dumps(metadaata))
        conn.close()

        # Initialize Redis stream reader
        reader = RedisStreamReader(redis_url=redis_url, stream_key=read_stream_key)
        # Initialize video generator and writer
        writer = None#RedisStreamWriter(redis_url, write_stream_key, shape=, maxlen=maxlen, fmt=fmt)
        for i,image in enumerate(reader.read_image_from_stream()):
            image,frame_metadaata = image_processor(i,image)
            frame_metadaata['shape']=image.shape
            if writer is None:
                writer = RedisStreamWriter(redis_url, write_stream_key, shape=image.shape, maxlen=maxlen, fmt=fmt)
            writer.write_to_stream(image,frame_metadaata)
        reader.close()
        writer.close()


    @staticmethod
    @celery_app.task(bind=True)
    def clone_stream(t: Task,redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='clone-stream:0',maxlen:int=10,fmt: str='.jpg',):
        
        image_processor=lambda i,image:(image, dict(count=i))
        metadaata=dict(task_id=t.request.id,video_src=read_stream_key,)
        CeleryTaskManager.stream2stream(image_processor=image_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key,maxlen=maxlen,fmt=fmt)
    @staticmethod
    @celery_app.task(bind=True)
    def flip_stream(t: Task,redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='flip-stream:0',maxlen:int=10,fmt: str='.jpg',):
        
        image_processor=lambda i,image:(cv2.flip(image, 1), dict(count=i))
        metadaata=dict(task_id=t.request.id,video_src=read_stream_key,)
        CeleryTaskManager.stream2stream(image_processor=image_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key,maxlen=maxlen,fmt=fmt)
    @staticmethod
    @celery_app.task(bind=True)
    def cv_resize_stream(t: Task,w:int,h:int, redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='resize-stream:0',maxlen:int=10,fmt: str='.jpg',):
        
        image_processor=lambda i,image:(cv2.resize(image, (w,h)), dict(count=i))
        metadaata=dict(task_id=t.request.id,video_src=read_stream_key,resize=(w,h))
        CeleryTaskManager.stream2stream(image_processor=image_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key,maxlen=maxlen,fmt=fmt)

    @staticmethod
    @celery_app.task(bind=True)
    def yolo_image_stream(t: Task, redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='ai-stream:0',maxlen=10,fmt='.jpg',
                                            modelname:str='yolov5s6',conf=0.6):
        
        model = torch.hub.load(f'ultralytics/{modelname[:6]}', modelname, pretrained=True)
        model.conf = conf
        model.eval()
        def image_processor(i,image,model=model):
            result = model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result.render()
            image = cv2.cvtColor(result.ims[0], cv2.COLOR_RGB2BGR)
            return image, dict(count=i)

        metadaata=dict(task_id=t.request.id,video_src=read_stream_key,
                        modelname=modelname,conf=conf,)
        
        CeleryTaskManager.stream2stream(image_processor=image_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key,maxlen=maxlen,fmt=fmt)