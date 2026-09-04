@echo off
setlocal
call conda activate qatarbird-afm-2026
if errorlevel 1 exit /b 1
python scripts\00_check_env.py || exit /b 1
python scripts\00_probe_birdmae.py --config configs\default.yaml || exit /b 1
python scripts\01_audit_kaggle.py --config configs\default.yaml --hash || exit /b 1
python scripts\02_build_kaggle_manifest.py --config configs\default.yaml || exit /b 1
REM Critical FIX2 step: decode MP3/M4A once, not repeatedly during every epoch.
python scripts\02b_cache_kaggle_audio.py --config configs\default.yaml --workers 2 || exit /b 1
python scripts\02c_check_split_coverage.py --config configs\default.yaml || exit /b 1
python scripts\03_train_kaggle_public.py --config configs\default.yaml --batch-size 32 --grad-accum 1 || exit /b 1
python scripts\04_evaluate_kaggle_public.py --config configs\default.yaml || exit /b 1
python scripts\05_transfer_report.py --config configs\default.yaml || exit /b 1
endlocal
