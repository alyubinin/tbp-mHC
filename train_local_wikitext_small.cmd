@echo off
REM cd /d "%~dp0"
cd /d e:\Dev\_code\alyubinin\tbp-mHC

echo Training mHC (small model, wikitext103 n=4)...
python train.py configN/small_model.py configN/with_mhc4.py configN/train_wikitext103.py
if errorlevel 1 exit /b 1

echo.
echo Training mHC_lite (small model, wikitext103 n=4)...
python train.py configN/small_model.py configN/with_mhc_lite4.py configN/train_wikitext103.py
if errorlevel 1 exit /b 1

echo.
echo Training KromHC (small model, wikitext103 n=4)...
python train.py configN/small_model.py configN/with_KromHC4.py configN/train_wikitext103.py
if errorlevel 1 exit /b 1

echo.
echo Training DORTBP2N (small model, wikitext103 n=4)...
python train.py configN/small_model.py configN/with_dortbp2n_mhc4.py configN/train_wikitext103.py
if errorlevel 1 exit /b 1

echo.
echo Done.
