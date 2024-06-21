import json
import os
import sys
from urllib.parse import urlparse
import time
from multiprocessing import shared_memory
import uuid

import cv2
import pandas as pd
import qrcode
import redis
import torch
import numpy as np
from PIL import Image

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
celery_app.conf.update(worker_pool_restarts=True)

def get_video_stream_info(redis_url: str = 'redis://127.0.0.1:6379'):
    url = urlparse(redis_url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    info = {}
    for k in conn.keys(f'info:*'):
        k = k.decode()
        if not conn.exists(k.replace('info:','')):
            continue
        info[k] = json.loads(conn.get(k))
    return info

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
    class Base:            
        def write(self,data):
            raise ValueError("[CommonIO.Reader]: This is Reader can not write")
        def read(self):
            raise ValueError("[CommonIO.Writer]: This is Writer can not read") 
        def close(self):
            raise ValueError("[CommonIO.Base]: not implemented")            
    class Reader(Base):
        def read(self):
            raise ValueError("[CommonIO.Reader]: not implemented")      
    class Writer(Base):
        def write(self,data):
            raise ValueError("[CommonIO.Writer]: not implemented")

class CommonStreamIO(CommonIO):
    class Base(CommonIO.Base):
        def write(self, data, metadata={}):
            raise ValueError("[CommonStreamIO.Reader]: This is Reader can not write")
        
        def read(self):
            raise ValueError("[CommonStreamIO.Writer]: This is Writer can not read") 
        
        def __iter__(self):
            return self

        def __next__(self):
            return self.read()
        
        def stop(self):
            raise ValueError("[StreamWriter]: not implemented")
        
    class StreamReader(CommonIO.Reader, Base):
        def read(self):
            return super().read(),{}
        
    class StreamWriter(CommonIO.Writer, Base):
        def write(self, data, metadata={}):
            raise ValueError("[StreamWriter]: not implemented")

class SharedMemoryIO(CommonIO):
    class Base(CommonIO.Base):
        def __init__(self,shm_name:str, create, array_shape, dtype=np.uint8):
            shm_name = shm_name.replace(':','_')
            shm_size = int(np.prod(array_shape) * np.dtype(dtype).itemsize)
            self.shm = shared_memory.SharedMemory(name=shm_name, create=create, size=shm_size)
            self.buffer = np.ndarray(array_shape, dtype=dtype, buffer=self.shm.buf)
            self.dtype = dtype
            self.array_shape = array_shape

        def close(self):
            self.shm.close()
        
    class Reader(CommonIO.Reader, Base):
        def read(self):
            return np.copy(self.buffer)

    class Writer(CommonIO.Writer, Base):
        def write(self, image:np.ndarray):
            if image.shape != self.array_shape:
                raise ValueError(f"Image does not match the predefined shape.({image.shape}!={self.array_shape})")
            if image.dtype != self.dtype:
                raise ValueError(f"Image does not match the predefined data type.({image.dtype}!={self.dtype})")
            np.copyto(self.buffer, image)
        
        def close(self):
            self.shm.close()
            self.shm.unlink()            

    def reader(self,shm_name:str, array_shape, dtype=np.uint8):
        return SharedMemoryIO.Reader(shm_name=shm_name, create=False, array_shape=array_shape, dtype=dtype)
    
    def writer(self,shm_name:str, array_shape, dtype=np.uint8):
        return SharedMemoryIO.Writer(shm_name=shm_name, create=True, array_shape=array_shape, dtype=dtype)

class RedisStream(CommonStreamIO):
    @staticmethod
    def getredis(redis_url):
        if type(redis_url) is not str:return redis_url
        url = urlparse(redis_url)
        return redis.Redis(host=url.hostname, port=url.port)

    @staticmethod
    def steam_list(redis_url) -> list:
        return [i.decode().replace('info:','') for i in RedisStream.getredis(redis_url).keys(f'info:*')]
    
    @staticmethod
    def stream_all_info(redis_url):
        conn = RedisStream.getredis(redis_url)
        info = {}
        for k in conn.keys(f'info:*'):
            k = k.decode()
            if not conn.exists(k.replace('info:','')):
                continue
            info[k] = json.loads(conn.get(k))
        return info

    @staticmethod
    def stream_exists(redis_url,stream_key) -> bool:
        conn = RedisStream.getredis(redis_url)
        return conn.exists(stream_key) and conn.exists(f'info:{stream_key}')       
    
    @staticmethod
    def steam_info_set(redis_url,stream_key,metadata={}):
        RedisStream.getredis(redis_url).set(f'info:{stream_key}',json.dumps(metadata))
    
    @staticmethod
    def steam_info_get(redis_url,stream_key):
        conn = RedisStream.getredis(redis_url)
        if RedisStream.stream_exists(conn,stream_key):
            return json.loads(conn.get(f'info:{stream_key}'))
        return {}
    
    @staticmethod
    def steam_info_delete(redis_url,stream_key):
        RedisStream.getredis(redis_url).delete(f'info:{stream_key}')
    
    @staticmethod
    def steam_stop(redis_url,stream_key):
        conn = RedisStream.getredis(redis_url)
        metadata = RedisStream.steam_info_get(conn,stream_key)
        metadata['stop'] = True
        RedisStream.steam_info_set(conn,stream_key,metadata)
        return {'msg':f'stop stream {stream_key}'}
    
    @staticmethod
    def steam_delete(redis_url,stream_key):
        conn = RedisStream.getredis(redis_url)
        conn.delete(stream_key)
        conn.delete(f'info:{stream_key}')
    
    class Base(CommonStreamIO.Base):
        def __init__(self, redis_url, stream_key, shape=None, maxlen=10, use_shared_memory=True):
            self.uuid = uuid.uuid4()
            self.conn = RedisStream.getredis(redis_url)
            self.redis_url = redis_url
            self.stream_key = stream_key
            self.shape = shape
            self.maxlen = maxlen
            self.use_shared_memory = use_shared_memory
            self._stop = False
            self.smIO:SharedMemoryIO.Base = None
            self.conn.set(f'{self.__class__.__name__}:{self.uuid}',json.dumps(dict(stream_key=stream_key, shape=shape)))

        def stop(self):
            self._stop = True

        def close(self):
            self.stop()

            if self.use_shared_memory and self.smIO is not None:
                self.smIO.close()
            self.conn.delete(self.stream_key)
            self.conn.delete(f'{self.__class__.__name__}:{self.uuid}')
            
            if RedisStream.stream_exists(self.conn,self.stream_key):
                RedisStream.steam_delete(self.conn,self.stream_key)
            self.conn.close()
    
        def get_steam_info(self):
            return RedisStream.steam_info_get(self.conn, self.stream_key)

        def set_steam_info(self,metadata):
            return RedisStream.steam_info_set(self.conn, self.stream_key, metadata)
        
    class StreamWriter(CommonStreamIO.StreamWriter, Base):
        def __init__(self, redis_url, stream_key, shape, metadata={}, maxlen=10, use_shared_memory=True):
            super().__init__(redis_url, stream_key, shape, maxlen, use_shared_memory)
            if self.use_shared_memory:
                self.smIO = SharedMemoryIO().writer(stream_key, shape)

            if stream_key is not None:
                if RedisStream.stream_exists(self.conn,stream_key):
                    raise ValueError(f'write stream key {stream_key} is already exists!')
            
            metadata['stop'] = False
            RedisStream.steam_info_set(self.conn,self.stream_key,metadata)
            self.metadata=metadata                            

        def write(self, image:np.ndarray, metadata={}):
            if image is not None:
                if image.shape != self.shape:raise ValueError(f'input shape{image.shape} is different of stream shap{self.shape}')
                metadata['shape']=np.asarray(image.shape,dtype=int).tobytes()
            message = metadata
            if self.use_shared_memory:
                self.smIO.write(image)
            else:
                metadata['image']=np.asarray(image,dtype=np.uint8).tobytes()
            self.conn.xadd(self.stream_key, message, maxlen=self.maxlen)

    class StreamReader(CommonStreamIO.StreamReader, Base):
        def __init__(self, redis_url, stream_key, use_shared_memory=True):
            super().__init__(redis_url, stream_key, None, None, use_shared_memory)
            self.last_id = '0-0'

            if stream_key is not None:
                if not RedisStream.stream_exists(self.conn,stream_key):
                    raise ValueError(f'read stream key {stream_key} is not exists!')
            
            # first run
            image,frame_metadata = self.read()
            self.shape = image.shape
            
        def read(self):
            if self._stop:
                raise StopIteration()
            messages = self.conn.xread({self.stream_key: self.last_id},count=1, block=1000)
            if len(messages)>0:
                for _, records in messages:
                    for record in records:
                        message_id, redis_metadata = record
                        if self.smIO is None and self.use_shared_memory:
                            self.smIO = SharedMemoryIO().reader(self.stream_key, np.frombuffer(redis_metadata[b'shape'], dtype=int))
                        self.last_id = message_id
                        if self.use_shared_memory:
                            image = self.smIO.read()
                        else:
                            shape = np.frombuffer(redis_metadata[b'shape'], dtype=int)
                            image = np.frombuffer(redis_metadata[b'image'], dtype=np.uint8)
                            image = image.reshape(shape)
                        return image,redis_metadata
            else:
                raise ValueError('no message from redis')

    def reader(self,redis_url, stream_key):
        return RedisStream.StreamReader(redis_url=redis_url, stream_key=stream_key)
    
    def writer(self, redis_url, write_stream_key, shape, metadata={}):
        return RedisStream.StreamWriter(redis_url, write_stream_key, shape=shape, metadata=metadata)
    
class VideoStreamReader(CommonStreamIO.StreamReader):

    @staticmethod
    def isFile(p):
        return not str(p).isdecimal()
    def isChrome(p):
        return str(p).lower()=='chrome'
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
            # first run
            image,frame_metadata = self.read()
            self.shape = image.shape

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

    class Chrome(Base):
        def __init__(self, fps=30, width=800, height=600):
            import pygetwindow as gw
            import pyautogui
            self.pyautogui = pyautogui
            chrome_windows = [window for window in gw.getWindowsWithTitle('Chrome')]
            if not chrome_windows:raise ValueError("No Chrome window found.")
            chrome_window = chrome_windows[0]
            left, top, right, bottom = chrome_window.left, chrome_window.top, chrome_window.right, chrome_window.bottom
            self.region = (left, top, right-left, bottom-top)
            self.fps = fps
            self.width = width
            self.height = height
            # first run
            image,frame_metadata = self.read()
            self.shape = image.shape
            
        def read(self):
            # Capture the region of the screen where Chrome is located
            img = self.pyautogui.screenshot(region=self.region)
            # Convert the image to a numpy array
            frame = np.array(img)            
            # Convert it from BGR (Pillow default) to RGB (OpenCV default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame,{}
        
    def reader(self, video_src=0, fps=30.0, width=800, height=600):
        if not VideoStreamReader.isFile(video_src):
            return VideoStreamReader.Camera(int(video_src),fps,width,height)
        if VideoStreamReader.isChrome(video_src):
            return VideoStreamReader.Chrome(fps,width,height)
        return VideoStreamReader.File(video_src,fps,width,height)

class RedisBidirectionalStream:
    class Bidirectional:
        def __init__(self, frame_processor=lambda i,frame,frame_metadata:(frame,frame_metadata),
                        stream_reader:RedisStream.StreamReader=None,stream_writer:RedisStream.StreamWriter=None):
            
            self.frame_processor = frame_processor
            self.stream_writer = stream_writer
            self.stream_reader = stream_reader
            self.streams:list[RedisStream.Base] = [self.stream_reader, self.stream_writer]
            
        def run(self):
            for s in self.streams:
                if s is None:raise ValueError('stream is None')
                
            res = {'msg':''}
            try:
                for frame_count,(image,frame_metadata) in enumerate(self.stream_reader):
                    if frame_count%100==0:
                        start_time = time.time()
                    else:
                        elapsed_time = time.time() - start_time + 1e-5
                        frame_metadata['fps'] = fps = (frame_count%100) / elapsed_time
                    

                    image,frame_processor_metadata = self.frame_processor(frame_count,image,frame_metadata)
                    frame_metadata.update(frame_processor_metadata)
                    self.stream_writer.write(image,frame_metadata)

                    if frame_count%1000==100:
                        metadata = self.stream_writer.get_steam_info()
                        if metadata.get('stop',False):     
                            for s in self.streams:
                                s.stop()
                            break
                        metadata['fps'] = fps
                        self.stream_writer.set_steam_info(metadata)

            except Exception as e:
                    res['error'] = str(e)
                    print(res)
            finally:
                for s in self.streams:
                    s.close()
                return res
    class WriteOnly(Bidirectional):
        def __init__(self, frame_processor=lambda i,frame,frame_metadata:(frame,frame_metadata),
                    stream_writer:RedisStream.StreamWriter=None):
            self.frame_processor = frame_processor
            self.stream_writer = stream_writer
            self.streams:list[RedisStream.Base] = [self.stream_writer]
            
            def mock_stream():
                while True:
                    yield None,{}
            self.stream_reader = mock_stream()        

    class ReadOnly(Bidirectional):
        def __init__(self, frame_processor=lambda i,frame,frame_metadata:(frame,frame_metadata),
                    stream_reader:RedisStream.StreamReader=None):
            
            self.frame_processor = frame_processor
            self.stream_reader = stream_reader

        def run(self):
            if self.stream_reader is None:raise ValueError('stream_reader is None')
            res = {'msg':''}
            try:
                for frame_count,(image,frame_metadata) in enumerate(self.stream_reader):
                    if frame_count%100==0:
                        start_time = time.time()
                    else:
                        elapsed_time = time.time() - start_time + 1e-5
                        frame_metadata['fps'] = fps = (frame_count%100) / elapsed_time

                    image,frame_metadata = self.frame_processor(frame_count,image,frame_metadata)

                    if frame_count%1000==100:
                        if self.stream_reader.get_steam_info().get('stop',False):
                            self.stream_reader.stop()
                            break
                        
            except Exception as e:
                    res['error'] = str(e)
                    print(res)
            finally:
                self.stream_reader.close()
                res['msg'] += f'\nstream {self.stream_reader.stream_key} reader.close()'
                return res
            
    def bidirectional(self, frame_processor,stream_reader:CommonStreamIO,stream_writer:RedisStream):
        return RedisBidirectionalStream.Bidirectional(frame_processor,stream_reader,stream_writer)
        
    def readOnly(self, frame_processor,stream_reader:RedisStream):
        return RedisBidirectionalStream.ReadOnly(frame_processor,stream_reader)
                
    def writeOnly(self, frame_processor,stream_writer:RedisStream):
        return RedisBidirectionalStream.WriteOnly(frame_processor,stream_writer)

class CeleryTaskManager:
    @staticmethod
    @celery_app.task(bind=True)
    def revoke(t: Task, task_id: str):
        """Method to revoke a task."""
        return celery_app.control.revoke(task_id, terminate=True)
    
    @staticmethod
    @celery_app.task(bind=True)    
    def stop_video_stream(t: Task, redis_stream_key='camera-stream:0', redis_url='redis://127.0.0.1:6379'):
        RedisStream.steam_stop(redis_url,redis_stream_key)
        return {'msg':f'stop stream {redis_stream_key}'}
    
    @staticmethod
    @celery_app.task(bind=True)    
    def stop_all_stream(t: Task, redis_url='redis://127.0.0.1:6379'):
        conn = RedisStream.getredis(redis_url)
        allks = RedisStream.steam_list(conn)
        for k in allks:
            RedisStream.steam_stop(conn,k)
        return {'msg':f'stop all stream {allks}'}

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
        
        stream_reader=VideoStreamReader().reader(video_src=video_src, fps=fps, width=width, height=height)
        def frame_gen(i,image,frame_metadata,stream_reader=stream_reader):            
            image,frame_metadata = stream_reader.read()
            # fps = frame_metadata.get('fps',0)
            # CeleryTaskManager.debug_cvshow(image.copy(),fps,f'Streamed directly from camera to {stream_key}')
            return image,frame_metadata
        metadata=dict(task_id=t.request.id,video_src=video_src, fps=fps,shape=stream_reader.shape)
        stream_writer=RedisStream().writer(redis_url,redis_stream_key,metadata=metadata,shape=stream_reader.shape)
        print(f"Stream {redis_stream_key} started. By task id of {t.request.id}")
        return RedisBidirectionalStream().writeOnly(frame_gen,stream_writer).run()
       
    @staticmethod
    @celery_app.task(bind=True)
    def cvshow_image_stream(t: Task, redis_url:str, stream_key:str, fontscale:float=1):

        def frame_processor(i,image,frame_metadata,stream_key=stream_key,fontscale=fontscale):
            fps = frame_metadata.get('fps',0)            
            res = CeleryTaskManager.debug_cvshow(image.copy(),fps,f'Streamed Image {stream_key}',fontscale)
            if not res:raise ValueError('Force to stop.')
            return None,{}

        try:
            stream_reader=RedisStream().reader(redis_url=redis_url, stream_key=stream_key)
            return RedisBidirectionalStream().readOnly(frame_processor,stream_reader).run()
        except Exception as e:
            cv2.destroyAllWindows()
            raise ValueError(str(e))

    @staticmethod
    def make_redis_bidirectional(frame_processor,metadata:dict,redis_url,read_stream_key,write_stream_key):        
        stream_reader=RedisStream().reader(redis_url=redis_url, stream_key=read_stream_key)
        
        outshape=frame_processor(0,(np.random.rand(*stream_reader.shape)*255).astype(np.uint8),{})[0].shape
        metadata.update(dict(video_src=read_stream_key,inshape=stream_reader.shape,outshape=outshape))

        stream_writer=RedisStream().writer(redis_url,write_stream_key,metadata=metadata,shape=outshape)
        return RedisBidirectionalStream().bidirectional(frame_processor,stream_reader,stream_writer)
    
    @staticmethod
    @celery_app.task(bind=True)
    def clone_stream(t: Task,redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='clone-stream:0'):
        
        frame_processor=lambda i,image,frame_metadata:(image, frame_metadata)
        metadata=dict(task_id=t.request.id)
        return CeleryTaskManager.make_redis_bidirectional(frame_processor,metadata,redis_url,read_stream_key,write_stream_key).run()
        
    @staticmethod
    @celery_app.task(bind=True)
    def flip_stream(t: Task,redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='flip-stream:0'):
        
        frame_processor=lambda i,image,frame_metadata:(cv2.flip(image, 1), frame_metadata)
        metadata=dict(task_id=t.request.id)
        return CeleryTaskManager.make_redis_bidirectional(frame_processor,metadata,redis_url,read_stream_key,write_stream_key).run()
        
    @staticmethod
    @celery_app.task(bind=True)
    def split_stream(t: Task,bbox, redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='flip-stream:0'):
        a,b,c,d = bbox
        frame_processor=lambda i,image,frame_metadata:(image[a:b,c:d], frame_metadata)
        metadata=dict(task_id=t.request.id,bbox=bbox)
        return CeleryTaskManager.make_redis_bidirectional(frame_processor,metadata,redis_url,read_stream_key,write_stream_key).run()

    @staticmethod
    @celery_app.task(bind=True)
    def cv_resize_stream(t: Task,w:int,h:int, redis_url: str, read_stream_key: str='camera-stream:0',
                                            write_stream_key: str='resize-stream:0'):
        
        def frame_processor(i,image,frame_metadata):
            return cv2.resize(image, (w,h)), frame_metadata
        metadata=dict(task_id=t.request.id,resize=(w,h))
        return CeleryTaskManager.make_redis_bidirectional(frame_processor,metadata,redis_url,read_stream_key,write_stream_key).run()

    @staticmethod
    @celery_app.task(bind=True)
    def _yolo_image_stream(t: Task, redis_url: str, read_stream_key: str='camera-stream:0',
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

        metadata=dict(task_id=t.request.id,modelname=modelname,conf=conf)
        return CeleryTaskManager.make_redis_bidirectional(frame_processor,metadata,redis_url,read_stream_key,write_stream_key).run()


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
            results = model(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results.render()
            image = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
            
            detections = results.pandas().xyxy[0]
            detections['confidence'] = pd.to_numeric(detections['confidence'], errors='coerce')

            # Step 2: Format and Filter Detection Results
            # Extract the top 10 detection results based on confidence
            top_detections = detections.nlargest(10, 'confidence')
            # Convert detection results to JSON
            detection_results = top_detections.to_dict(orient='records')
            detection_json = json.dumps(detection_results, indent=4)
            # Step 3: Generate the QR Code
            qr = qrcode.QRCode(
                version=1,  # Adjust version if needed
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(detection_json)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")
            qr_img = qr_img.resize((200, 200))  # Adjust size if needed
            qr_img = cv2.cvtColor(np.asarray(qr_img,dtype=np.uint8)*255, cv2.COLOR_GRAY2RGB)
            # Step 4: Overlay QR Code on Detection Image
            offset = 10
            image[offset:qr_img.shape[0]+offset,offset:qr_img.shape[1]+offset,:] = qr_img
            return image, frame_metadata

        metadata=dict(task_id=t.request.id,modelname=modelname,conf=conf)
        return CeleryTaskManager.make_redis_bidirectional(frame_processor,metadata,redis_url,read_stream_key,write_stream_key).run()