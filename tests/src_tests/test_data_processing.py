
"""Tests that the prepare.py script produces the correct files"""

import pytest, sys
from pathlib import Path
sys.path.insert(1, str((Path(__file__).parent.parent.parent).resolve()))     #path of the project working directory relative to this file
from src.prepare import process_dataset

work_dir = Path.cwd()
data_preproc_script = work_dir / 'src' / 'prepare.py'
processed_dataset_path = work_dir / 'data' / 'processed'    # with Path / is equivalent to os.path.join
processed_training_set_path = processed_dataset_path / 'processed_training_set.csv' 
processed_val_set_path = processed_dataset_path / 'processed_validation_set.csv' 
proc_test_set_path = processed_dataset_path / 'processed_test_set.csv'
y_train_path = processed_dataset_path / 'y_train.csv'
y_val_path = processed_dataset_path / 'y_validation.csv'

@pytest.mark.processing
def test_files_creation():
    process_dataset()         #takes 1h 
    assert processed_dataset_path.is_dir()
    assert processed_training_set_path.is_file()
    assert processed_val_set_path.is_file()
    assert proc_test_set_path.is_file()
    assert y_train_path.is_file()
    assert y_val_path.is_file()