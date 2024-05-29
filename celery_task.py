import base64
import json
import os
import random
import subprocess
import sys
from pathlib import Path
import time
from typing import List
import cv2
import open3d as o3d
import numpy as np
# from scipy.spatial.transform import Rotation as R

from .calib import SurveyTargetsDetector
from .PointCloud import Pillar, PointCloud, plane_equation    
from celery import Celery
from celery.app import task as Task

####################################################################################
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')
IP = '192.168.1.80'
# os.environ.setdefault('CELERY_BROKER_URL', 'redis://'+IP)
# os.environ.setdefault('CELERY_RESULT_BACKEND', 'redis://'+IP+'/0')
os.environ.setdefault('CELERY_BROKER_URL', f'amqp://guest@{IP}//')
os.environ.setdefault('CELERY_RESULT_BACKEND', f'amqp://guest@{IP}//')
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
# Import the Celery application
from celery_app import app
import cv2
import redis
import time
from urllib.parse import urlparse

class SimpleMovingAverage(object):
    def __init__(self, value=0.0, count=7):
        self.count = int(count)
        self.current = float(value)
        self.samples = [self.current] * self.count

    def add(self, value):
        v = float(value)
        self.samples.insert(0, v)
        o = self.samples.pop()
        self.current = self.current + (v-o)/self.count

class Video:
    def __init__(self, infile=0, fps=30.0):
        self.isFile = not str(infile).isdecimal()
        self.ts = time.time()
        self.infile = infile
        self.cam = cv2.VideoCapture(self.infile)
        if not self.isFile:
            self.cam.set(cv2.CAP_PROP_FPS, fps)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        else:
            self.fps = self.cam.get(cv2.CAP_PROP_FPS)
            self.sma = SimpleMovingAverage(value=0.1, count=19)

    def __iter__(self):
        return self

    def __next__(self):
        ret_val, img0 = self.cam.read()
        if not ret_val:
            raise StopIteration()
        img = cv2.flip(img0, 1) if not self.isFile else img0
        return img

class CeleryTaskManager:    
    @staticmethod
    @celery_app.task(bind=True)
    def revoke(t: Task, task_id: str):
        """Method to revoke a task."""
        return celery_app.control.revoke(task_id, terminate=True)
    
    @staticmethod
    @celery_app.task(bind=True)    
    def capture_video(t: Task, infile, fps, output, url, fmt='.jpg', maxlen=1000, count=None):
        url = urlparse(url)
        conn = redis.Redis(host=url.hostname, port=url.port)
        loader = Video(infile=infile, fps=fps)
        for i, img in enumerate(loader):
            if count is not None and i >= count:
                break
            _, data = cv2.imencode(fmt, img)
            msg = {
                'count': i,
                'image': data.tobytes()
            }
            _id = conn.xadd(output, msg, maxlen=maxlen)
            print('frame:', i, 'id:', _id)

