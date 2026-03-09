@echo off
setlocal enabledelayedexpansion

set "CMAKE_GENERATOR=Visual Studio 17 2022"

python -m build --wheel --outdir "%SRC_DIR%\dist\wheels" --no-isolation --skip-dependency-check
if %ERRORLEVEL% neq 0 exit 1

for %%f in ("%SRC_DIR%\dist\wheels\*.whl") do (
    python -m pip install "%%f" --no-deps --no-build-isolation --prefix "%PREFIX%" -vv
    if !ERRORLEVEL! neq 0 exit 1
)

