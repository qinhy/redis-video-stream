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

class CeleryTaskManager:    
    @staticmethod
    @celery_app.task(bind=True)
    def revoke(t: Task, task_id: str):
        """Method to revoke a task."""
        return celery_app.control.revoke(task_id, terminate=True)
    
    @staticmethod
    @celery_app.task(bind=True)
