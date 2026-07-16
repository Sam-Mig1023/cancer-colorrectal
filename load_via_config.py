
import h5py
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
h5_path = 'models/mobilenetv2_base_only.h5'

with h5py.File(h5_path, 'r') as f:
    model_config = json.loads(f.attrs['model_config'])
    model_json = json.dumps(model_config)

print("Creating model from config without building...")
# Try to create model with custom code to avoid building


# Temporarily override the Sequential.add method to skip _maybe_rebuild()
original_add = Sequential.add

def patched_add(self, layer):
    # Skip the _maybe_rebuild() and other logic
    self.built = False
    self.layers.append(layer)
    # Update layer names
    if hasattr(layer, '_name'):
        self._layer_names.add(layer._name)
    else:
        self._layer_names.add(layer.name)

try:
    Sequential.add = patched_add
    # Use model_from_json
    model = keras.models.model_from_json(model_json)
    print("Model created from config!")

    # Now load the weights
    model.load_weights(h5_path)
    print("Weights loaded!")

    # Now test prediction
    dummy_input = tf.random.normal([1, 224, 224, 3])
    pred = model.predict(dummy_input, verbose=0)
    print(f"Prediction successful! Shape: {pred.shape}")

    # Save fixed model
    model.save('models/mobilenetv2_base_only_fixed.h5')
    print("Fixed model saved!")
finally:
    # Restore original method
    Sequential.add = original_add

# Test reloading
print("\nTesting reloading fixed model...")
fixed_model = keras.models.load_model('models/mobilenetv2_base_only_fixed.h5')
fixed_pred = fixed_model.predict(dummy_input, verbose=0)
print("✓ Fixed model loaded and predicted!")
