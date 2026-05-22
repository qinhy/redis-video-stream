@echo off
pushd "%~dp0\..\.."
uv run uvicorn redis_video_stream.api:app --host 0.0.0.0
popd
