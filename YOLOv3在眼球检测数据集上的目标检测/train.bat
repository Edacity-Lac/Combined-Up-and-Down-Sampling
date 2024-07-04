@echo off

cd E:\\wyk\\yolov3
:: 假设所有的Python程序都在同一目录下，或者你可以指定完整的文件路径

:: 执行程序1
python train.py   test3/log01

:: 检查程序1是否成功执行（这里只是一个简单的示例，你可能需要根据你的程序来定制检查逻辑）
if %errorlevel% neq 0 (
    echo 程序1执行失败，退出脚本。
    exit /b %errorlevel%
)

:: 执行程序2
python train.py   test3/log02
if %errorlevel% neq 0 (
    echo 程序2执行失败，退出脚本。
    exit /b %errorlevel%
)

:: 类似地，继续执行其他程序...
python train.py   test3/log03
if %errorlevel% neq 0 (
    echo 程序3执行失败，退出脚本。
    exit /b %errorlevel%
)

python train.py   test3/log04
if %errorlevel% neq 0 (
    echo 程序4执行失败，退出脚本。
    exit /b %errorlevel%
)

python train.py   test3/log05
if %errorlevel% neq 0 (
    echo 程序5执行失败，退出脚本。
    exit /b %errorlevel%
)

echo 所有程序都已成功执行。