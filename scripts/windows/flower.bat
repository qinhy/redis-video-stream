@echo off
pushd "%~dp0\..\.."
set FLOWER_UNAUTHENTICATED_API=true
uv run celery -A redis_video_stream.tasks flower -b redis://127.0.0.1 --loglevel=INFO
popd
