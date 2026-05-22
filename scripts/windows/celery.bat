@echo off
pushd "%~dp0\..\.."
uv run celery -A redis_video_stream.tasks worker -l info -P threads
popd
