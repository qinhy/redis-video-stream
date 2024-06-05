import json
import os
import sys
from urllib.parse import urlparse
import time
from multiprocessing import shared_memory

import cv2
import redis
import torch
import numpy as np

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

    def read_from_stream(self):
        while True:
            yield next(self),{}

    def __iter__(self):
        return self

    def __next__(self):
        ret_val, img = self.cam.read()
        if not ret_val:
            raise StopIteration()
        return cv2.flip(img, 1) if not self.isFile else img

    def close(self):
        del self.cam
    
class SharedMemoryStreamWriter:
    def __init__(self, shm_name, array_shape, dtype=np.uint8):
        shm_name = shm_name.replace(':','_')
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
        shm_name = shm_name.replace(':','_')
        self.array_shape = array_shape
        self.dtype = dtype
        # Attach to the existing shared memory
        self.shm = shared_memory.SharedMemory(name=shm_name)
        # Map the numpy array to the shared memory
        self.buffer = np.ndarray(array_shape, dtype=dtype, buffer=self.shm.buf)

    def read_from_sharedMemory(self):
        # Simply return a copy of the array to ensure the data remains consistent
        return np.copy(self.buffer)

    def close(self):
        self.shm.close()


class RedisStreamWriter:
    def __init__(self, redis_url, stream_key, shape, maxlen=10):
        self.conn = getredis(redis_url)
        self.stream_key = stream_key
        self.maxlen = maxlen
        self.smwriter = SharedMemoryStreamWriter(stream_key, shape)

    def write_to_stream(self, image, metadata={}):
        if image is not None:
            metadata['shape']=np.asarray(image.shape,dtype=int).tobytes()
        message = metadata
        
        self.smwriter.write_to_stream(image)
        self.conn.xadd(self.stream_key, message, maxlen=self.maxlen)

    def close(self):
        self.smwriter.close()

class RedisStreamReader:
    def __init__(self, redis_url, stream_key):
        self.conn = getredis(redis_url)
        self.stream_key = stream_key
        self.smreader = None

    def read_from_redis_stream(self,count=1):
        last_id = '0-0'
        while True:
            messages = self.conn.xread({self.stream_key: last_id},count=count, block=1000)
            if messages:
                for _, records in messages:
                    for record in records:
                        message_id, redis_metadata = record
                        if self.smreader is None:
                            self.smreader = SharedMemoryStreamReader(self.stream_key,
                                                                    np.frombuffer(redis_metadata[b'shape'], dtype=int))
                        last_id = message_id  # Update last_id to the latest message_id read
                        yield redis_metadata

    def read_from_stream(self,count=1):
        for redis_metadata in self.read_from_redis_stream(count=count):
            image = self.smreader.read_from_sharedMemory()
            yield image,redis_metadata
    
    def close(self):
        self.smreader.close()

class CeleryTaskManager:    
    @staticmethod
    @celery_app.task(bind=True)
    def revoke(t: Task, task_id: str):
        """Method to revoke a task."""
        return celery_app.control.revoke(task_id, terminate=True)

    @staticmethod
    def _stop_stream(redis_stream_key='camera-stream:0', redis_url='redis://127.0.0.1:6379'):
        conn = getredis(redis_url)
        info = conn.get(f'info:{redis_stream_key}')
        if info is None:
            return {'msg':'no such stream'}
        info = json.loads(info)
        conn.delete(redis_stream_key)
        conn.delete(f'info:{redis_stream_key}')
        return info
    
    @staticmethod
    @celery_app.task(bind=True)    
    def stop_video_stream(t: Task, redis_stream_key='camera-stream:0', redis_url='redis://127.0.0.1:6379'):
        info = CeleryTaskManager._stop_stream(redis_stream_key,redis_url)
        if 'task_id' in info:
            celery_app.control.revoke(info['task_id'], terminate=True)
        return {'msg':f'delete stream {redis_stream_key}'}
    
    @staticmethod
    @celery_app.task(bind=True)    
    def stop_all_stream(t: Task, redis_url='redis://127.0.0.1:6379'):
        conn = getredis(redis_url)
        allks = conn.keys(f'info:*')
        conn.close()
        for k in allks:
            redis_stream_key = k.decode().replace('info:','')
            info = CeleryTaskManager._stop_stream(redis_stream_key,redis_url)
            if 'task_id' in info:
                celery_app.control.revoke(info['task_id'], terminate=True)
        return {'msg':f'delete all stream {allks}'}

    @staticmethod
    def stream2stream(image_processor=lambda i,image,redis_metadata:(image,redis_metadata),
                      redis_url: str='redis://127.0.0.1:6379',read_stream_key: str='camera-stream:0',
                      write_stream_key: str='out-stream:0',metadaata={},
                      stream_reader=None,stream_writer=None):
        
        conn = getredis(redis_url)
        
        if read_stream_key is not None:
            if not is_stream_exists(conn,read_stream_key):
                raise ValueError(f'read stream key {read_stream_key} is not exists!')
        
        if write_stream_key is not None:
            if is_stream_exists(conn,write_stream_key):                
                raise ValueError(f'write stream key {write_stream_key} is already exists!')
            
            conn.set(f'info:{write_stream_key}',json.dumps(metadaata))

        conn.close()

        # Initialize Redis stream reader
        reader = stream_reader if stream_reader is not None else RedisStreamReader(redis_url=redis_url, stream_key=read_stream_key)
        # Initialize video generator and writer
        writer = stream_writer if stream_writer is not None else None

        start_time = time.time()
        try:
            for frame_count,(image,redis_metadata) in enumerate(reader.read_from_stream()):
                
                elapsed_time = time.time() - start_time
                redis_metadata['fps']= frame_count / elapsed_time if elapsed_time > 0 else 0
                
                image,frame_metadaata = image_processor(frame_count,image,redis_metadata)

                if frame_metadaata is not None:
                    redis_metadata.update(frame_metadaata)

                if writer:
                    writer.write_to_stream(image,redis_metadata)
                elif write_stream_key is not None:
                    writer = RedisStreamWriter(redis_url, write_stream_key, shape=image.shape)
                    
        
        finally:
            if reader is not None:
                reader.close()
            if write_stream_key is not None:
                if writer is not None:
                    writer.close()                
                CeleryTaskManager._stop_stream(write_stream_key,redis_url)
                return {'msg':f'delete stream {write_stream_key}'}

    @staticmethod
    def debug_cvshow(image,fps,title,fontscale=1.0):
        cv2.putText(image, f"FPS: {fps:.2f}", (int(10*fontscale), int(30*fontscale)), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(title, image)
        if cv2.waitKey(1) == 27:  # ESC key
            cv2.destroyWindow(title)
            return False
        return True
            
    @staticmethod
    @celery_app.task(bind=True)    
    def start_video_stream(t: Task, video_src:str, fps:float, width=800, height=600,
                                    redis_stream_key='camera-stream:0', redis_url='redis://127.0.0.1:6379'):
        
        def image_processor(i,image,redis_metadata,stream_key=redis_stream_key):
            # fps = redis_metadata.get('fps',0)
            # CeleryTaskManager.debug_cvshow(image.copy(),fps,f'Streamed directly from camera to {stream_key}')
            return image,redis_metadata
        stream_reader=VideoGenerator(video_src=video_src, fps=fps, width=width, height=height)

        print(f"Stream {redis_stream_key} started. By task id of {t.request.id}")
        metadaata=dict(task_id=t.request.id,video_src=video_src, fps=fps, width=width, height=height)
        return CeleryTaskManager.stream2stream(image_processor=image_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=None,
                                    write_stream_key=redis_stream_key,
                                    stream_reader=stream_reader)
       
    @staticmethod
    @celery_app.task(bind=True)
    def cvshow_image_stream(t: Task, redis_url:str, stream_key:str, fontscale:float=1):

        def image_processor(i,image,redis_metadata,stream_key=stream_key,fontscale=fontscale):
            fps = redis_metadata.get('fps',0)            
            res = CeleryTaskManager.debug_cvshow(image.copy(),fps,f'Streamed Image {stream_key}',fontscale)
            if not res:raise ValueError('Force to stop.')
            return None,None

        metadaata=dict(task_id=t.request.id,video_src=stream_key)
        try:
            return CeleryTaskManager.stream2stream(image_processor=image_processor,metadaata=metadaata,
                                        redis_url=redis_url,read_stream_key=stream_key,
                                        write_stream_key=None)
        except Exception as e:
            cv2.destroyAllWindows()
            raise ValueError(str(e))
        
    @staticmethod
    @celery_app.task(bind=True)
    def clone_stream(t: Task,redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='clone-stream:0'):
        
        image_processor=lambda i,image,redis_metadata:(image, redis_metadata)
        metadaata=dict(task_id=t.request.id,video_src=read_stream_key)
        return CeleryTaskManager.stream2stream(image_processor=image_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key)
    @staticmethod
    @celery_app.task(bind=True)
    def flip_stream(t: Task,redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='flip-stream:0'):
        
        image_processor=lambda i,image,redis_metadata:(cv2.flip(image, 1), redis_metadata)
        metadaata=dict(task_id=t.request.id,video_src=read_stream_key)
        return CeleryTaskManager.stream2stream(image_processor=image_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key)
    @staticmethod
    @celery_app.task(bind=True)
    def split_stream(t: Task,bbox, redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='flip-stream:0'):
        a,b,c,d = bbox
        image_processor=lambda i,image,redis_metadata:(image[a:b,c:d], redis_metadata)
        metadaata=dict(task_id=t.request.id,video_src=read_stream_key,bbox=bbox)
        return CeleryTaskManager.stream2stream(image_processor=image_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key)
    @staticmethod
    @celery_app.task(bind=True)
    def cv_resize_stream(t: Task,w:int,h:int, redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='resize-stream:0'):
        
        image_processor=lambda i,image,redis_metadata:(cv2.resize(image, (w,h)), redis_metadata)
        metadaata=dict(task_id=t.request.id,video_src=read_stream_key,resize=(w,h))
        return CeleryTaskManager.stream2stream(image_processor=image_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key)

    @staticmethod
    @celery_app.task(bind=True)
    def yolo_image_stream(t: Task, redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='ai-stream:0',
                                            modelname:str='yolov5s6',conf=0.6):
        
        model = torch.hub.load(f'ultralytics/{modelname[:6]}', modelname, pretrained=True)
        model.conf = conf
        model.eval()
        def image_processor(i,image,redis_metadata,model=model):
            result = model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result.render()
            image = cv2.cvtColor(result.ims[0], cv2.COLOR_RGB2BGR)
            return image, redis_metadata

        metadaata=dict(task_id=t.request.id,video_src=read_stream_key,
                        modelname=modelname,conf=conf)
        
        return CeleryTaskManager.stream2stream(image_processor=image_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key)