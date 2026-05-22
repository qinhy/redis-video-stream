@echo off
pushd "%~dp0\..\.."
uv run python -m redis_video_stream.ui
popd
