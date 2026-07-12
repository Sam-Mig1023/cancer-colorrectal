
import h5py
import json
import tensorflow as tf
from tensorflow import keras

h5_path = 'models/mobilenetv2_base_only.h5'
fixed_path = 'models/mobilenetv2_base_only_fixed.h5'

# Step 1: Read config
with h5py.File(h5_path, 'r') as f:
    model_config = json.loads(f.attrs['model_config'])

# Step 2: Remove the first layer (InputLayer)
original_layers = model_config['config']['layers']
print(f"Original number of layers: {len(original_layers)}")
print(f"Removing layer 0: {original_layers[0]['class_name']}")

# Remove InputLayer (index 0)
new_layers = original_layers[1:]
model_config['config']['layers'] = new_layers
print(f"New number of layers: {len(new_layers)}")

# Step 3: Update build_input_shape if needed (though Sequential doesn't need it)
model_config['config'].pop('build_input_shape', None)

# Step 4: Save modified config to a temporary json
temp_json = 'temp_model_config.json'
with open(temp_json, 'w') as f:
    json.dump(model_config, f)
print(f"Modified config saved to {temp_json}")

# Step 5: Load model from modified config
print("\nLoading model from modified config...")
model = keras.models.model_from_json(json.dumps(model_config))

# Step 6: Load weights from original h5
print("Loading original model weights...")
model.load_weights(h5_path, by_name=True, skip_mismatch=True)

# Step 7: Test prediction
print("\nTesting prediction...")
dummy_input = tf.random.normal([1, 224, 224, 3])
pred = model.predict(dummy_input, verbose=0)
print(f"Prediction successful! Shape: {pred.shape}, Sum: {pred.sum():.4f}")

# Step 8: Save fixed model
model.save(fixed_path)
print(f"\nFixed model saved to {fixed_path}!")

# Step 9: Test reloading fixed model
print("\nTesting reloading fixed model...")
fixed_model = keras.models.load_model(fixed_path)
fixed_pred = fixed_model.predict(dummy_input, verbose=0)
print("✓ Fixed model loaded and predicted successfully!")
print(f"  Fixed prediction shape: {fixed_pred.shape}, Sum: {fixed_pred.sum():.4f}")
