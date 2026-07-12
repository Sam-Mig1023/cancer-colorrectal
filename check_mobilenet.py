
import tensorflow as tf
from tensorflow import keras

path = 'models/mobilenetv2_base_only.h5'

try:
    # Try to load with compile=False
    model = keras.models.load_model(path, compile=False)
    print("Model type:", type(model))
    print("\nModel config:")
    import json
    print(json.dumps(model.get_config(), indent=2))
except Exception as e:
    print("Error loading model with compile=False:", e)
    import traceback
    print("\nTraceback:", traceback.format_exc())
