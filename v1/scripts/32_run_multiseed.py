import argparse,subprocess,sys,yaml,tempfile
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--seeds',type=int,nargs='+',default=[42,123,2026]); p.add_argument('--stages',nargs='+',default=['ap1','ap2','ap3']); a=p.parse_args(); root=Path(__file__).resolve().parents[1]; base=yaml.safe_load(open(root/a.config,encoding='utf-8')) if not Path(a.config).is_absolute() else yaml.safe_load(open(a.config,encoding='utf-8'))
for seed in a.seeds:
 cfg=dict(base); cfg['project']=dict(base['project']); cfg['project']['seed']=seed
 path=root/'configs'/f'_seed_{seed}.yaml'; yaml.safe_dump(cfg,open(path,'w',encoding='utf-8'),sort_keys=False)
 try:
  if 'ap1' in a.stages: subprocess.run([sys.executable,str(root/'scripts/20_train_ap1.py'),'--config',str(path),'--variant','A5_full_acpsar'],check=True,cwd=root)
  if 'ap2' in a.stages: subprocess.run([sys.executable,str(root/'scripts/21_train_ap2.py'),'--config',str(path),'--variant','B5_full_uaps'],check=True,cwd=root)
  if 'ap3' in a.stages: subprocess.run([sys.executable,str(root/'scripts/22_train_ap3.py'),'--config',str(path),'--variant','C5_full_dcgs'],check=True,cwd=root)
 finally:
  path.unlink(missing_ok=True)
