@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Activate the pinned conda env only when that machine's conda exists;
rem otherwise keep whatever environment the caller already activated.
if exist "C:\Users\nolovr\miniconda3\condabin\conda.bat" call C:\Users\nolovr\miniconda3\condabin\conda.bat activate isaaclab-sonic >nul 2>&1

rem Isaac Sim location: prefer the repo-local _isaac_sim (per-machine
rem symlink/junction, not committed); fall back to the original machine's
rem install only when the local one is absent.
set "ISAAC_SIM_PATH=D:\reboot\isaac-sim"
if exist "%~dp0_isaac_sim\" set "ISAAC_SIM_PATH=%~dp0_isaac_sim"

rem Locate the omni.usd.libs extension without pinning its version hash
rem (the hash differs across Isaac Sim builds/machines).
set "USD_LIBS_PATH="
for /d %%i in ("%ISAAC_SIM_PATH%\extscache\omni.usd.libs-*") do if not defined USD_LIBS_PATH set "USD_LIBS_PATH=%%~fi"

rem Add DLL paths to PATH (Windows backslash format)
set "PATH=%ISAAC_SIM_PATH%\kit;%PATH%"
if defined USD_LIBS_PATH set "PATH=%USD_LIBS_PATH%\bin;%USD_LIBS_PATH%\bin\usd;%PATH%"

rem Add pxr to PYTHONPATH (site only exists in binary installs; harmless otherwise)
set "PYTHONPATH=%ISAAC_SIM_PATH%\site;%PYTHONPATH%"
if defined USD_LIBS_PATH set "PYTHONPATH=%USD_LIBS_PATH%;%PYTHONPATH%"

rem Run Python
python.exe %*
exit /b %ERRORLEVEL%
