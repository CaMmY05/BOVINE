@echo off
setlocal enabledelayedexpansion

:: Set paths
set BASE_DIR=%~dp0
set SCRIPTS_DIR=%BASE_DIR%scripts
set DATA_DIR=%BASE_DIR%data
set MODELS_DIR=%BASE_DIR%models\classification

:: Set parameters
set BATCH_SIZE=32
set NUM_EPOCHS=50
set LEARNING_RATE=0.001
set PATIENCE=5

:: Create output directories
if not exist "%MODELS_DIR%\resnet18_cow_v1" mkdir "%MODELS_DIR%\resnet18_cow_v1"
if not exist "%MODELS_DIR%\resnet18_buffalo_v1" mkdir "%MODELS_DIR%\resnet18_buffalo_v1"

:: Train Cow model
echo ========================================
echo Training ResNet-18 for Cows
echo ========================================
python "%SCRIPTS_DIR%\prepare_resnet_data.py" ^
    --src_dir "%DATA_DIR%\final_organized\cows" ^
    --output_dir "%DATA_DIR%\resnet_cows" ^
    --test_size 0.15 ^
    --val_size 0.15 ^
    --seed 42

python "%SCRIPTS_DIR%\train_resnet18.py" ^
    --data_dir "%DATA_DIR%\resnet_cows" ^
    --output_dir "%MODELS_DIR%\resnet18_cow_v1" ^
    --batch_size %BATCH_SIZE% ^
    --num_epochs %NUM_EPOCHS% ^
    --lr %LEARNING_RATE% ^
    --patience %PATIENCE%

:: Train Buffalo model
echo ========================================
echo Training ResNet-18 for Buffaloes
echo ========================================
python "%SCRIPTS_DIR%\prepare_resnet_data.py" ^
    --src_dir "%DATA_DIR%\final_organized\buffaloes" ^
    --output_dir "%DATA_DIR%\resnet_buffaloes" ^
    --test_size 0.15 ^
    --val_size 0.15 ^
    --seed 42

python "%SCRIPTS_DIR%\train_resnet18.py" ^
    --data_dir "%DATA_DIR%\resnet_buffaloes" ^
    --output_dir "%MODELS_DIR%\resnet18_buffalo_v1" ^
    --batch_size %BATCH_SIZE% ^
    --num_epochs %NUM_EPOCHS% ^
    --lr %LEARNING_RATE% ^
    --patience %PATIENCE%

echo ========================================
echo Training complete!
echo ========================================
pause
