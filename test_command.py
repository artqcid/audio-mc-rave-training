import shlex
from pathlib import Path

BASE_DIR = Path(__file__).parent
preprocessed_path = '/mnt/c/Users/marku/Documents/GitHub/artqcid/ai-projects/audio-mc-rave-training/Rave Training Test/processed'
config_file = 'raspberry.gin'
epochs = 3
rave_data_path = preprocessed_path

# Create Python training script - write to file directly to avoid heredoc issues
# Based on successful manual test: need gin search path and dataset sr/n_signal args
train_script = f'''import gin
gin.add_config_file_search_path("/mnt/c/Users/marku/Documents/GitHub/artqcid/ai-projects/audio-mc-rave-training/.venv/lib/python3.12/site-packages/rave")
gin.parse_config_file("configs/{config_file}")

from rave import RAVE, dataset
import pytorch_lightning as pl
import torch

model = RAVE()
train_dataset = dataset.get_dataset("{shlex.quote(rave_data_path)}", sr=44100, n_signal=16384)
trainer = pl.Trainer(max_epochs={epochs}, gpus=1 if torch.cuda.is_available() else 0)
trainer.fit(model, train_dataset)
'''

script_path = '/tmp/train_rave.py'
command = (
    f'bash -lc "cd {shlex.quote(str(BASE_DIR))} && source .venv/bin/activate && '
    f'python3 -c {shlex.quote(train_script)} > {shlex.quote(script_path)} && '
    f'python {shlex.quote(script_path)}"'
)

print('Generated command:')
print(command)
print()
print('---')
print('Train script that will be written:')
print(train_script)