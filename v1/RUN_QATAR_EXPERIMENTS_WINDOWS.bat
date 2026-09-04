@echo off
call conda activate qatarbird-afm-2026
python scripts\20_train_ap1.py --config configs\default.yaml --variant A5_full_acpsar
if errorlevel 1 exit /b 1
python scripts\21_train_ap2.py --config configs\default.yaml --variant B5_full_uaps
if errorlevel 1 exit /b 1
python scripts\22_train_ap3.py --config configs\default.yaml --variant C5_full_dcgs
if errorlevel 1 exit /b 1
python scripts\30_aggregate_results.py --config configs\default.yaml
