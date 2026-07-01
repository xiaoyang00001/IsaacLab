@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Activate conda environment
call C:\Users\nolovr\miniconda3\condabin\conda.bat activate isaaclab-sonic >nul 2>&1

rem Get ISAAC SIM path (actual location, not symlink)
set "ISAAC_SIM_PATH=D:\reboot\isaac-sim"

rem Get the omni.usd.libs path
set "USD_LIBS_PATH=%ISAAC_SIM_PATH%\extscache\omni.usd.libs-1.0.1+69cbf6ad.wx64.r.cp311"

rem Add DLL paths to PATH (Windows backslash format)
set "PATH=%USD_LIBS_PATH%\bin;%USD_LIBS_PATH%\bin\usd;%ISAAC_SIM_PATH%\kit;%PATH%"

rem Add pxr to PYTHONPATH
set "PYTHONPATH=%USD_LIBS_PATH%;%ISAAC_SIM_PATH%\site;%PYTHONPATH%"

rem Run Python
python.exe %*
exit /b %ERRORLEVEL%
