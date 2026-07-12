
import h5py
import json
import tensorflow as tf
from tensorflow import keras

h5_path = 'models/mobilenetv2_base_only.h5'
fixed_path = 'models/mobilenetv2_base_only_fixed.h5'

# Step 1: Read config and weights
with h5py.File(h5_path, 'r') as f:
    model_config = json.loads(f.attrs['model_config'])

# Let's create the model without auto-building
# Try creating as Functional instead of Sequential
print("Original model type:", model_config['class_name'])

# Let's try to load the model with a custom build that bypasses the check
# Or let's try to load weights directly into a manually constructed model
print("Trying manual model construction...")

# Manually build the correct model
input_layer = keras.Input(shape=(224, 224, 3), name='input_1')
# Use MobileNetV2 from keras.applications
base_model = keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet',
    input_tensor=input_layer
)
x = keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(base_model.output)
x = keras.layers.Dense(256, activation='relu', name='dense')(x)
x = keras.layers.Dropout(0.5, name='dropout')(x)
output_layer = keras.layers.Dense(9, activation='softmax', name='dense_1')(x)
manual_model = keras.Model(inputs=input_layer, outputs=output_layer)

print("Manual model created! Now loading weights from original h5...")

# Load weights
manual_model.load_weights(h5_path, by_name=True, skip_mismatch=True)

print("Weights loaded! Testing prediction...")
dummy_input = tf.random.normal([1, 224, 224, 3])
pred = manual_model.predict(dummy_input, verbose=0)
print(f"Prediction shape: {pred.shape}, sum: {pred.sum()}")

# Save fixed model
manual_model.save(fixed_path)
print(f"Fixed model saved to {fixed_path}!")

# Test loading the fixed model
print("\nTesting fixed model load...")
test_loaded = keras.models.load_model(fixed_path)
print("✓ Fixed model loaded successfully!")
test_pred = test_loaded.predict(dummy_input, verbose=0)
print(f"✓ Test prediction shape: {test_pred.shape}")
