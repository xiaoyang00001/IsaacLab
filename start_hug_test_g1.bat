@echo off
chcp 65001 >nul
setlocal
REM G1 抱箱演示启动脚本（Windows 版，对应 Ubuntu 上的 ./start_teleop_g1.sh --hug-test）
REM
REM 用法:
REM   start_hug_test_g1.bat                    按默认配置启动
REM   start_hug_test_g1.bat --free-root        额外参数原样透传给 hug_box_g1_test.py
REM
REM 覆盖默认值（示例，先 set 再运行）:
REM   set GR00T_WBC_ROOT=D:\GR00T-WholeBodyControl
REM   start_hug_test_g1.bat

set "SCRIPT_DIR=%~dp0"

REM ---------- 单机独立测试配置 ----------
REM 本机固定当 1 号机器人 + 箱子 publisher 角色，保证 test_box 是动力学刚体。
REM （若按 robot_2/subscriber 启动，test_box 是 kinematic 跟随体，物理上抱不住，
REM   脚本会直接拒绝运行。）
if not defined ISAACLAB_LOCAL_ROBOT_ID set "ISAACLAB_LOCAL_ROBOT_ID=1"
if not defined ISAACLAB_OBJECT_SYNC_ROLE set "ISAACLAB_OBJECT_SYNC_ROLE=publisher"

REM G1 43-DoF USD 所在的 GR00T-WholeBodyControl 仓库。
REM 未设置时代码会自动尝试 F:\ISAACWholeBody\GR00T-WholeBodyControl 等路径。
REM if not defined GR00T_WBC_ROOT set "GR00T_WBC_ROOT=F:\ISAACWholeBody\GR00T-WholeBodyControl"

if not defined TASK set "TASK=Isaac-PickPlace-Locomanipulation-G1-Abs-v0"
if not defined DEVICE set "DEVICE=cuda:0"

echo [start_hug_test_g1] ISAACLAB_LOCAL_ROBOT_ID   = %ISAACLAB_LOCAL_ROBOT_ID%
echo [start_hug_test_g1] ISAACLAB_OBJECT_SYNC_ROLE = %ISAACLAB_OBJECT_SYNC_ROLE%
echo [start_hug_test_g1] GR00T_WBC_ROOT            = %GR00T_WBC_ROOT%
echo [start_hug_test_g1] TASK / DEVICE             = %TASK% / %DEVICE%
echo [start_hug_test_g1] 抱箱演示模式：全自动六阶段，约 2 分钟

call "%SCRIPT_DIR%isaaclab.bat" -p scripts\environments\teleoperation\hug_box_g1_test.py --device %DEVICE% --task %TASK% %*

endlocal
