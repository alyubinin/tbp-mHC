@echo off
REM cd /d "%~dp0"
cd /d e:\Dev\_code\alyubinin\tbp-mHC

echo Training mHC (small model, shakespeare_char n=32)...
REM python train.py configN/small_model.py configN/with_mhc32.py configN/local_shakespeare_n8.py
if errorlevel 1 exit /b 1

echo.
echo Training KromHC (small model, shakespeare_char n=32)...
python train.py configN/small_model.py configN/with_KromHC32.py configN/local_shakespeare_n8.py
if errorlevel 1 exit /b 1

echo.
echo Training DORTBP2N (small model, shakespeare_char n=32)...
python train.py configN/small_model.py configN/with_dortbp2n_mhc32.py configN/local_shakespeare_n8.py
if errorlevel 1 exit /b 1

echo.
echo Done.
