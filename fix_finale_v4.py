import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py

# Load old model
old = keras.models.load_model('BPLD_CNN_model.h5', compile=False)

# Save weights layer by layer into h5py directly (no Keras metadata)
with h5py.File('weights_raw.h5', 'w') as f:
    for i, layer in enumerate(old.layers):
        w = layer.get_weights()
        if len(w) > 0:
            grp = f.create_group(f'layer_{i}')
            for j, arr in enumerate(w):
                grp.create_dataset(f'w{j}', data=arr)
            print(f"Saved layer {i} ({layer.name}): {[arr.shape for arr in w]}")

print("\n✅ Saved weights_raw.h5 — upload this to Google Drive")