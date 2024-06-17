import json
import os
import sys
from urllib.parse import urlparse
import time
from multiprocessing import shared_memory
import uuid

import cv2
import redis
import torch
import numpy as np

from celery import Celery
from celery.app import task as Task

####################################################################################
IP = '127.0.0.1'
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')
os.environ.setdefault('CELERY_BROKER_URL', 'redis://'+IP)
os.environ.setdefault('CELERY_RESULT_BACKEND', 'redis://'+IP+'/0')
####################################################################################
celery_app = Celery('tasks')
celery_app.conf.update(
    worker_pool_restarts=True,
)

def getredis(redis_url):
    url = urlparse(redis_url)
    return redis.Redis(host=url.hostname, port=url.port)

def get_video_stream_info(redis_url: str = 'redis://127.0.0.1:6379'):
    conn = getredis(redis_url)
    info = {}
    for k in conn.keys(f'info:*'):
        k = k.decode()
        if not conn.exists(k.replace('info:','')):
            continue
        info[k] = json.loads(conn.get(k))
    return info

def is_stream_exists(conn,stream_key):
    return conn.exists(stream_key) and conn.exists(f'info:{stream_key}')

def WrappTask(task:Task):
    def update_progress_state(progress=1.0,msg=''):
        task.update_state(state='PROGRESS',meta={'progress': progress,'msg':msg})
        task.send_event('task-progress', result={'progress': progress})
        
    def update_error_state(error='null'):
        task.update_state(state='FAILURE',meta={'error': error})
    
    task.progress = update_progress_state
    task.error = update_error_state
    return task 

class CommonIO:
    class State:
        class Base:            
            def write(self,data):
                raise ValueError("[CommonIO.State.Reader]: This is Reader can not write")
            def read(self):
                raise ValueError("[CommonIO.State.Writer]: This is Writer can not read") 
            def close(self):
                raise ValueError("[CommonIO.State.Base]: not implemented")            
        class Reader(Base):
            def read(self):
                raise ValueError("[CommonIO.State.Reader]: not implemented")      
        class Writer(Base):                        
            def write(self,data):
                raise ValueError("[CommonIO.State.Writer]: not implemented")
    
    def __init__(self, state:State.Base=None):
        self.state = state

    def read(self):
        if self.state is None: ValueError("[CommonIO.read]: not init Reader state")
        return self.state.read()
    
    def write(self, data):
        if self.state is None: ValueError("[CommonIO.read]: not init Writer state")
        return self.state.write(data)
    
    def close(self):
        if self.state is None: ValueError("[CommonIO.read]: not init state") 
        self.state.close()

class CommonStreamIO(CommonIO):
    class State(CommonIO.State):
        class Base(CommonIO.State.Base):

            def write(self, data, metadata={}):
                raise ValueError("[CommonStreamIO.State.Reader]: This is Reader can not write")
            
            def read(self):
                raise ValueError("[CommonStreamIO.State.Writer]: This is Writer can not read") 
            
            def __iter__(self):
                return self

            def __next__(self):
                return self.read()
            
            def stop(self):
                raise ValueError("[StreamWriter]: not implemented")
            
        class StreamReader(CommonIO.State.Reader, Base):
            def read(self):
                return super().read(),{}
            
        class StreamWriter(CommonIO.State.Writer, Base):
            def write(self, data, metadata={}):
                raise ValueError("[StreamWriter]: not implemented")
            
    def __iter__(self):
        return self.state
    
    def __next__(self):
        return self.state.__next__()
    
    def write(self, image, metadata={}):
        return self.state.write(image, metadata)

class SharedMemoryIO(CommonIO):
    class State(CommonIO.State):
        class Base(CommonIO.State.Base):
            def __init__(self,shm_name:str, create, array_shape, dtype=np.uint8):
                shm_name = shm_name.replace(':','_')
                shm_size = int(np.prod(array_shape) * np.dtype(dtype).itemsize)
                self.shm = shared_memory.SharedMemory(name=shm_name, create=create, size=shm_size)
                self.buffer = np.ndarray(array_shape, dtype=dtype, buffer=self.shm.buf)
                self.dtype = dtype
                self.array_shape = array_shape

            def close(self):
                self.shm.close()
            
        class Reader(CommonIO.State.Reader, Base):
            def read(self):
                return np.copy(self.buffer)

        class Writer(CommonIO.State.Writer, Base):
            def write(self, image:np.ndarray):
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

    def reader(self,shm_name:str, create, array_shape, dtype=np.uint8):
        self.state = SharedMemoryIO.State.Reader(shm_name=shm_name, create=create, array_shape=array_shape, dtype=dtype)
        return self
    
    def writer(self,shm_name:str, create, array_shape, dtype=np.uint8):
        self.state = SharedMemoryIO.State.Writer(shm_name=shm_name, create=create, array_shape=array_shape, dtype=dtype)
        return self

class RedisStream(CommonStreamIO):
    class State(CommonStreamIO.State):
        class Base(CommonStreamIO.State.Base):
            def __init__(self, redis_url, stream_key, shape=None, maxlen=10, use_shared_memory=True):
                self.uuid = uuid.uuid4()
                self.conn = getredis(redis_url)
                self.redis_url = redis_url
                self.stream_key = stream_key
                self.shape = shape
                self.maxlen = maxlen
                self.use_shared_memory = use_shared_memory
                self._stop = False
                self.smIO:SharedMemoryIO = None
                self.conn.set(f'{self.__class__.__name__}:{self.uuid}',json.dumps(dict(stream_key=stream_key, shape=shape)))

            def stop(self):
                self._stop = True

            def close(self):
                if self.use_shared_memory and self.smIO is not None:
                    self.smIO.close()
                self.conn.delete(self.stream_key)
                self.conn.delete(f'{self.__class__.__name__}:{self.uuid}')
                self.conn.close()

        class StreamWriter(CommonStreamIO.State.StreamWriter, Base):
            def __init__(self, redis_url, stream_key, shape, maxlen=10, use_shared_memory=True):
                super().__init__(redis_url, stream_key, shape, maxlen, use_shared_memory)
                if self.use_shared_memory:
                    self.smIO = SharedMemoryIO().writer(stream_key, True, shape)

            def write(self, image:np.ndarray, metadata={}):
                if image is not None:
                    metadata['shape']=np.asarray(image.shape,dtype=int).tobytes()
                message = metadata
                if self.use_shared_memory:
                    self.smIO.write(image)
                else:
                    metadata['image']=np.asarray(image,dtype=np.uint8).tobytes()
                self.conn.xadd(self.stream_key, message, maxlen=self.maxlen)

        class StreamReader(CommonStreamIO.State.StreamReader, Base):
            def __init__(self, redis_url, stream_key, use_shared_memory=True):
                super().__init__(redis_url, stream_key, None, None, use_shared_memory)
                self.last_id = '0-0'

            def read(self):
                if self._stop:
                    raise StopIteration()
                messages = self.conn.xread({self.stream_key: self.last_id},count=1, block=1000)
                if len(messages)>0:
                    for _, records in messages:
                        for record in records:
                            message_id, redis_metadata = record
                            if self.smIO is None and self.use_shared_memory:
                                self.smIO = SharedMemoryIO().reader(self.stream_key, False, np.frombuffer(redis_metadata[b'shape'], dtype=int))
                            self.last_id = message_id  # Update last_id to the latest message_id read
                            if self.use_shared_memory:
                                image = self.smIO.read()
                            else:
                                shape = np.frombuffer(redis_metadata[b'shape'], dtype=int)
                                image = np.frombuffer(redis_metadata[b'image'], dtype=np.uint8)
                                # print(shape,image.shape)
                                image = image.reshape(shape)
                            return image,redis_metadata
                else:
                    raise ValueError('no message from redis')

    def reader(self,redis_url, stream_key):
        self.state = RedisStream.State.StreamReader(redis_url=redis_url, stream_key=stream_key)
        return self
    
    def writer(self, redis_url, write_stream_key, shape):
        self.state = RedisStream.State.StreamWriter(redis_url, write_stream_key, shape=shape)
        return self

class VideoStreamReader(CommonStreamIO.State.StreamReader):
    class State:
        @staticmethod
        def isFile(p):
            return not str(p).isdecimal()
        class Base:
            def __init__(self, video_src=0, fps=30.0, width=800, height=600):
                self.video_src=video_src
                self.cam = cv2.VideoCapture(self.video_src)
                self.cam.set(cv2.CAP_PROP_FPS, fps)
                self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                self.fps = self.cam.get(cv2.CAP_PROP_FPS)
                self.width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH,)
                self.height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

            def read(self):
                ret_val, img = self.cam.read()
                if not ret_val:raise StopIteration()
                return img
            
            def close(self):
                del self.cam
            
        class Camera(Base):
            def read(self):
                return cv2.flip( super().read() , 1),{}

        class File(Base):
            def read(self):
                return super().read(),{}

        
    def __init__(self, video_src=0, fps=30.0, width=800, height=600):
        if not VideoStreamReader.State.isFile(video_src):
            self.state = VideoStreamReader.State.Camera(int(video_src),fps,width,height)
        else:
            self.state = VideoStreamReader.State.File(video_src,fps,width,height)
        
        self.fps = self.state.fps
        self.width = self.state.width
        self.height = self.state.height

    def close(self):
        self.state.close()

    def read(self):
        return self.state.read()
    
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
        conn = getredis(redis_url)
        metadaata = conn.get(f'info:{redis_stream_key}')
        metadaata = json.loads(metadaata)
        metadaata['stop'] = True
        conn.set(f'info:{redis_stream_key}',json.dumps(metadaata))
        # info = CeleryTaskManager._stop_stream(redis_stream_key,redis_url)
        # if 'task_id' in info:
        #     celery_app.control.revoke(info['task_id'], terminate=True)
        return {'msg':f'stop stream {redis_stream_key}'}
    
    @staticmethod
    @celery_app.task(bind=True)    
    def stop_all_stream(t: Task, redis_url='redis://127.0.0.1:6379'):
        conn = getredis(redis_url)
        allks = conn.keys(f'info:*')
        conn.close()
        for k in allks:
            redis_stream_key = k.decode().replace('info:','')
            metadaata = conn.get(f'info:{redis_stream_key}')
            metadaata = json.loads(metadaata)
            metadaata['stop'] = True
            conn.set(f'info:{redis_stream_key}',json.dumps(metadaata))
            # info = CeleryTaskManager._stop_stream(redis_stream_key,redis_url)
            # if 'task_id' in info:
            #     celery_app.control.revoke(info['task_id'], terminate=True)
        return {'msg':f'stop all stream {allks}'}

    @staticmethod
    def stream2stream(frame_processor=lambda i,image,frame_metadata:(image,frame_metadata),
                      redis_url: str='redis://127.0.0.1:6379',read_stream_key: str='camera-stream:0',
                      write_stream_key: str='out-stream:0',metadaata={},
                      stream_reader:CommonStreamIO=None,stream_writer=None):
        
        metadaata['stop'] = False
        
        conn = getredis(redis_url)
        
        if read_stream_key is not None:
            if not is_stream_exists(conn,read_stream_key):
                raise ValueError(f'read stream key {read_stream_key} is not exists!')
        
        if write_stream_key is not None:
            if is_stream_exists(conn,write_stream_key):
                raise ValueError(f'write stream key {write_stream_key} is already exists!')
            
            conn.set(f'info:{write_stream_key}',json.dumps(metadaata))

        res = {'msg':''}
        # Initialize Redis stream reader
        reader:CommonStreamIO = stream_reader if stream_reader else RedisStream().reader(redis_url=redis_url, stream_key=read_stream_key)
        # Initialize video generator and writer
        writer = stream_writer if stream_writer else None
            
        try:
            # first run
            image,frame_metadata = next(reader)
            image,frame_metadaata = frame_processor(0,image,frame_metadata)
            
            if write_stream_key and writer is None:
                writer = RedisStream().writer(redis_url, write_stream_key, shape=image.shape)

            for frame_count,(image,frame_metadata) in enumerate(reader):

                if frame_count%100==0:
                    start_time = time.time()
                else:
                    elapsed_time = time.time() - start_time + 1e-5
                    metadaata['fps'] = frame_metadata['fps'] = (frame_count%100) / elapsed_time

                image,frame_metadaata = frame_processor(frame_count,image,frame_metadata)

                if frame_metadaata:
                    frame_metadata.update(frame_metadaata)

                if writer:
                    writer.write(image,frame_metadata)

                if write_stream_key and frame_count%1000==100:
                    fps,metadaata = metadaata['fps'],json.loads(conn.get(f'info:{write_stream_key}'))
                    metadaata['fps'] = fps
                    if metadaata.get('stop',True):
                        if reader and hasattr(reader,'stop'):reader.stop()
                        if writer and hasattr(writer,'stop'):writer.stop()
                        break
                    conn.set(f'info:{write_stream_key}',json.dumps(metadaata))

        except Exception as e:
                res['error'] = str(e)

        finally:
            conn.close()
            if reader and hasattr(reader,'close'):
                reader.close()
                res['msg'] += f'\nstream {write_stream_key} reader.close()'
            
            if writer and hasattr(writer,'close'):
                writer.close()
                res['msg'] += f'\stream {write_stream_key} writer.close()'

            if write_stream_key is not None:
                CeleryTaskManager._stop_stream(write_stream_key,redis_url)
                res['msg'] += f'\ndelete stream {write_stream_key}'
                
            return res

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
        
        def frame_processor(i,image,frame_metadata,stream_key=redis_stream_key):
            # fps = frame_metadata.get('fps',0)
            # CeleryTaskManager.debug_cvshow(image.copy(),fps,f'Streamed directly from camera to {stream_key}')
            return image,frame_metadata
        stream_reader=VideoStreamReader(video_src=video_src, fps=fps, width=width, height=height)

        print(f"Stream {redis_stream_key} started. By task id of {t.request.id}")
        metadaata=dict(task_id=t.request.id,video_src=video_src, fps=fps, width=width, height=height)
        return CeleryTaskManager.stream2stream(frame_processor=frame_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=None,
                                    write_stream_key=redis_stream_key,
                                    stream_reader=stream_reader)
       
    @staticmethod
    @celery_app.task(bind=True)
    def cvshow_image_stream(t: Task, redis_url:str, stream_key:str, fontscale:float=1):

        def frame_processor(i,image,frame_metadata,stream_key=stream_key,fontscale=fontscale):
            fps = frame_metadata.get('fps',0)            
            res = CeleryTaskManager.debug_cvshow(image.copy(),fps,f'Streamed Image {stream_key}',fontscale)
            if not res:raise ValueError('Force to stop.')
            return None,None

        metadaata=dict(task_id=t.request.id,video_src=stream_key)
        try:
            return CeleryTaskManager.stream2stream(frame_processor=frame_processor,metadaata=metadaata,
                                        redis_url=redis_url,read_stream_key=stream_key,
                                        write_stream_key=None)
        except Exception as e:
            cv2.destroyAllWindows()
            raise ValueError(str(e))
        
    @staticmethod
    @celery_app.task(bind=True)
    def clone_stream(t: Task,redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='clone-stream:0'):
        
        frame_processor=lambda i,image,frame_metadata:(image, frame_metadata)
        metadaata=dict(task_id=t.request.id,video_src=read_stream_key)
        return CeleryTaskManager.stream2stream(frame_processor=frame_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key)
    @staticmethod
    @celery_app.task(bind=True)
    def flip_stream(t: Task,redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='flip-stream:0'):
        
        frame_processor=lambda i,image,frame_metadata:(cv2.flip(image, 1), frame_metadata)
        metadaata=dict(task_id=t.request.id,video_src=read_stream_key)
        return CeleryTaskManager.stream2stream(frame_processor=frame_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key)
    @staticmethod
    @celery_app.task(bind=True)
    def split_stream(t: Task,bbox, redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='flip-stream:0'):
        a,b,c,d = bbox
        frame_processor=lambda i,image,frame_metadata:(image[a:b,c:d], frame_metadata)
        metadaata=dict(task_id=t.request.id,video_src=read_stream_key,bbox=bbox)
        return CeleryTaskManager.stream2stream(frame_processor=frame_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key)
    @staticmethod
    @celery_app.task(bind=True)
    def cv_resize_stream(t: Task,w:int,h:int, redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='resize-stream:0'):
        
        frame_processor=lambda i,image,frame_metadata:(cv2.resize(image, (w,h)), frame_metadata)
        metadaata=dict(task_id=t.request.id,video_src=read_stream_key,resize=(w,h))
        return CeleryTaskManager.stream2stream(frame_processor=frame_processor,metadaata=metadaata,
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
        model((np.random.rand(1280,1280,3)*255).astype(np.uint8)) # for pre loading

        def frame_processor(i,image,frame_metadata,model=model):
            result = model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result.render()
            image = cv2.cvtColor(result.ims[0], cv2.COLOR_RGB2BGR)
            return image, frame_metadata

        metadaata=dict(task_id=t.request.id,video_src=read_stream_key,
                        modelname=modelname,conf=conf)
        
        return CeleryTaskManager.stream2stream(frame_processor=frame_processor,metadaata=metadaata,
                                    redis_url=redis_url,read_stream_key=read_stream_key,
                                    write_stream_key=write_stream_key)