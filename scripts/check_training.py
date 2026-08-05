import sys
import os
sys.path.insert(0, os.getcwd())
import app
from fastapi.testclient import TestClient

client = TestClient(app.app)
print('cwd', os.getcwd())
print('processed exists', os.path.isdir('processed'))
print('processed files', len([f for f in os.listdir('processed') if os.path.isfile(os.path.join('processed', f))]))
resp = client.post(
    '/api/train',
    data={
        'model_name': 'rave_mini',
        'preprocessed_path': 'processed',
        'model_output_path': 'trained_models',
        'batch_size': 1,
        'epochs': 1,
        'learning_rate': 0.001,
        'use_rave': 'false',
        'latent_size': 16,
    },
)
print('status', resp.status_code)
print('text', resp.text)
