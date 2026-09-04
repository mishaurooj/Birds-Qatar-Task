"""Runs AP1/AP2/AP3 variants sequentially. Long BirdMAE run: use intentionally."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--stage',choices=['ap1','ap2','ap3','all'],default='all'); p.add_argument('--epochs',type=int); a=p.parse_args(); root=Path(__file__).resolve().parents[1]
cmds=[]
if a.stage in ['ap1','all']:
 for v in ['A0_frozen_linear','A1_single_adapter','A2_sparse_router','A3_router_condition','A4_router_prototype','A5_full_acpsar']:
  cmds.append([sys.executable,str(root/'scripts/20_train_ap1.py'),'--config',a.config,'--variant',v])
if a.stage in ['ap2','all']:
 for v in ['B0_5s_sigmoid','B1_10s_sigmoid','B2_adaptive_multiscale','B3_set_decoder','B4_set_intervals','B5_full_uaps']:
  cmds.append([sys.executable,str(root/'scripts/21_train_ap2.py'),'--config',a.config,'--variant',v])
if a.stage in ['ap3','all']:
 for v in ['C0_global','C1_static_hard','C2_static_soft','C3_dynamic_soft','C4_dynamic_contrastive','C5_full_dcgs']:
  cmds.append([sys.executable,str(root/'scripts/22_train_ap3.py'),'--config',a.config,'--variant',v])
for c in cmds:
 if a.epochs: c+=['--epochs',str(a.epochs)]
 print('\nRUN:', ' '.join(c)); subprocess.run(c,check=True,cwd=root)
subprocess.run([sys.executable,str(root/'scripts/30_aggregate_results.py'),'--config',a.config],check=True,cwd=root)
