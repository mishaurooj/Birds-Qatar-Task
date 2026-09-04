import sys, platform
print('Python:', sys.version)
print('Platform:', platform.platform())
try:
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('CUDA runtime:', torch.version.cuda)
    if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
    x=torch.randn(4,4)
    if torch.cuda.is_available(): x=x.cuda(); print('CUDA tensor test:', float((x@x).mean().cpu()))
    else: print('CPU tensor test:', float((x@x).mean()))
except Exception as e:
    print('\nPYTORCH IMPORT/LOAD FAILED:', repr(e))
    print('If this is WinError 182/fbgemm.dll, delete the old qatarbird-afm env and create environment_gpu_windows.yml from this package.')
    print('Also install/update Microsoft Visual C++ 2015-2022 Redistributable (x64), reboot, then recreate the environment.')
    raise
import torchaudio, transformers, pandas, sklearn, soundfile, librosa
print('torchaudio:', torchaudio.__version__)
print('transformers:', transformers.__version__)
print('pandas:', pandas.__version__)
print('scikit-learn:', sklearn.__version__)
print('Environment check passed.')
