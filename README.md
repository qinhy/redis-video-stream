# redis-video-stream
To wrap your video capture and processing script into a Celery task for improved stability and scalability, you will need to do a few things:

1. **Set up a Celery application**: Configure Celery with Redis as the broker and optionally as the result backend.
2. **Convert the main video processing loop into a Celery task**: This will allow you to distribute this task over multiple workers if needed.
3. **Handle initialization and cleanup**: Ensure that your Celery tasks properly manage resources such as opening and closing connections to Redis and video capture devices.

Here’s an example of how you might refactor your script to use Celery:

### Step 1: Install Celery and Redis
If not already installed, you will need Celery and Redis. You can install Celery with Redis support via pip:

```bash
pip install celery redis
```

### Step 2: Set Up the Celery Application
Create a new Python file for your Celery configuration. Let’s call it `celery_app.py`:

```python
from celery import Celery

app = Celery('video_tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

app.conf.task_routes = {'video_capture.capture_video': {'queue': 'video'}}
```

### Step 3: Refactor Your Video Processing Code into a Celery Task
Modify your video processing script to define a Celery task. Here's how you might refactor the code:

```python
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

@app.task
def capture_video(infile, fps, output, url, fmt='.jpg', maxlen=10000, count=None):
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

if __name__ == '__main__':
    # Example usage: launch a task
    capture_video.delay(webcam, 15.0, 'camera:0', 'redis://127.0.0.1:6379')
```

### Step 4: Start Celery Worker
Run a Celery worker that will execute the video capture tasks:

```bash
celery -A celery_app worker --loglevel=info --queues=video
```

### Final Notes
- **Task Invocation**: You can invoke the `capture_video` task from any part of your application using `capture_video.delay()` with appropriate arguments.
- **Resource Management**: Make sure the webcam and Redis connections are properly closed if needed. This example assumes a continuous run; add error handling and cleanup where necessary.
- **Concurrency**: Depending on your workload, you might need to adjust the concurrency settings of your Celery worker.

This setup allows you to decouple video capture from processing and scale by adding more workers or distributing tasks across multiple machines.
