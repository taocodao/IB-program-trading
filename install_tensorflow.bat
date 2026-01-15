@echo off
echo ========================================
echo TensorFlow Installation Script
echo ========================================
echo.

echo Checking Python version...
python --version
echo.

echo Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Installing TensorFlow...
python -m pip install tensorflow
echo.

echo Verifying installation...
python -c "import tensorflow as tf; print('SUCCESS: TensorFlow version:', tf.__version__)"
echo.

echo ========================================
echo Installation Complete
echo ========================================
pause
