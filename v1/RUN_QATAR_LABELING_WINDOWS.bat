@echo off
call conda activate qatarbird-afm-2026
python scripts\10_audit_qatar.py --config configs\default.yaml
python scripts\11_prepare_qatar_label_clips.py --config configs\default.yaml
call conda deactivate
call conda activate qatarbird-birdnet
python scripts\12_birdnet_label_qatar.py --config configs\default.yaml
call conda deactivate
call conda activate qatarbird-afm-2026
python scripts\13_build_qatar_manifests.py --config configs\default.yaml
python scripts\14_analyze_qatar_labels.py --config configs\default.yaml
