
import h5py
import json
import tensorflow as tf
from tensorflow import keras

h5_path = 'models/mobilenetv2_base_only.h5'

with h5py.File(h5_path, 'r') as f:
    model_config = json.loads(f.attrs['model_config'])
    layers_config = model_config['config']['layers']
    print(f"Number of layers in config: {len(layers_config)}")
    print("\nLayers list:")
    for i, layer in enumerate(layers_config):
        print(f"  Layer {i}: {layer['class_name']} (name: {layer['config'].get('name', 'N/A')})")

    mobilenet_config = layers_config[1]  # Index 1 is MobileNetV2
    print("\nMobileNetV2 layer config (index 1):")
    print(json.dumps(mobilenet_config, indent=2))

    # Try to create just the MobileNetV2 layer
    mobilenet_layer = keras.layers.deserialize(mobilenet_config)
    print("\nMobileNetV2 layer created!")

    # Test what it outputs
    dummy_input = tf.random.normal([1, 224, 224, 3])
    output = mobilenet_layer(dummy_input)
    print(f"Output type: {type(output)}")
    if isinstance(output, (list, tuple)):
        print(f"Number of outputs: {len(output)}")
        for i, o in enumerate(output):
            print(f"  Output {i} shape: {o.shape}")
    else:
        print(f"Output shape: {output.shape}")
