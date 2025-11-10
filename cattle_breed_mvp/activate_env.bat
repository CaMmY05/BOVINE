@echo off
echo ================================================================================
echo   CATTLE BREED DETECTION MVP - Environment Activation
echo ================================================================================
echo.
echo Activating virtual environment...
echo.

cd /d "%~dp0"
call ..\cattle_mvp_env\Scripts\activate.bat

echo.
echo ================================================================================
echo   Environment activated!
echo ================================================================================
echo.
echo Quick commands:
echo   - Verify setup: python scripts/verify_setup.py
echo   - Prepare data: python scripts/prepare_data.py
echo   - Train model:  python scripts/train_classifier.py
echo   - Launch app:   streamlit run app.py
echo.
echo For detailed instructions, see SETUP_INSTRUCTIONS.txt
echo ================================================================================
echo.
