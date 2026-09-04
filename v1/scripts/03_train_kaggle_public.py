# Compatibility wrapper for the filtered public dataset training stage.
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('03_train_kaggle114.py')), run_name='__main__')
