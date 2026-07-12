
import tensorflow as tf
from tensorflow import keras
import traceback

print("Testing each model...\n")

models_to_test = [
    ('CNN Simple', 'models/cnn_simple_model.h5'),
    ('MobileNetV2 Base', 'models/mobilenetv2_base_only.h5'),
    ('Hybrid Attention', 'models/Fast_HybridAttention_final.h5'),
    ('Hybrid Autoencoder', 'models/Fast_HybridAutoencoder_final.h5'),
    ('ResNet50V2', 'models/resnet50_model'),
]

for name, path in models_to_test:
    print(f"\n=== {name} ===")
    try:
        if name == 'ResNet50V2':
            model = keras.layers.TFSMLayer(
                path,
                call_endpoint='serving_default'
            )
            # Try a dummy pass
            dummy_input = tf.random.normal([1, 224, 224, 3])
            outputs = model(dummy_input)
            print("ResNet50V2 outputs:", outputs)
        else:
            model = keras.models.load_model(path)
            model.summary()
            # Try a dummy prediction
            dummy_input = tf.random.normal([1, 224, 224, 3])
            pred = model.predict(dummy_input, verbose=0)
            print("Prediction shape:", pred.shape)
    except Exception as e:
        print(f"Error loading {name}: {e}")
        print("Traceback:\n", traceback.format_exc())
