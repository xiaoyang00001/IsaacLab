@echo off
call C:\Users\nolovr\miniconda3\condabin\conda.bat activate isaaclab-pin >nul 2>&1
python %*
exit /b %ERRORLEVEL%
