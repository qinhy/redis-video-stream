@echo off
pushd "%~dp0\..\.."
timeout /t 5 >NUL
uv run python examples/pipeline.py
popd
