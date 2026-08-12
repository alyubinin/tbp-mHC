@echo off
REM cd /d "%~dp0"
cd /d e:\Dev\_code\alyubinin\tbp-mHC

echo Training mHC (medium model, shakespeare_char n=16)...
python train.py configN/medium_model.py configN/with_mhc16.py configN/local_shakespeare_n8.py
if errorlevel 1 exit /b 1

echo.
echo Training KromHC (medium model, shakespeare_char n=16)...
python train.py configN/medium_model.py configN/with_KromHC16.py configN/local_shakespeare_n8.py
if errorlevel 1 exit /b 1

echo Training DORTBP2N (medium model, shakespeare_char n=16)...
python train.py configN/medium_model.py configN/with_dortbp2n_mhc16.py configN/local_shakespeare_n8.py
if errorlevel 1 exit /b 1

echo.
echo Done.
