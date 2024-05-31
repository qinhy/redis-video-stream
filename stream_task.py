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
    def __init__(self, video_src=0, fps=30.0):
        self.isFile = not str(video_src).isdecimal()
        self.ts = time.time()
        self.video_src = video_src
        self.cam = cv2.VideoCapture(self.video_src)
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

@app.task
def capture_video(video_src, fps, output, url, fmt='.jpg', maxlen=10000, count=None):
    url = urlparse(url)
    conn = redis.Redis(host=url.hostname, port=url.port)
    loader = Video(video_src=video_src, fps=fps)
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

if __name__ == '__main__':
    # Example usage: launch a task
    capture_video.delay(webcam, 15.0, 'camera:0', 'redis://127.0.0.1:6379')
