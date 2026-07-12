
from app import load_models_and_confusion_matrices
import traceback
import tensorflow as tf

print("Testing load_models_and_confusion_matrices() from app.py...\n")
try:
    models, conf, roc = load_models_and_confusion_matrices()
    print("Models loaded successfully!")
    print("Loaded models:", list(models.keys()))

    # Test all models
    print("\nTesting all models...")
    for name, model in models.items():
        print(f"\n--- Testing {name} ---")
        if name in ['Hybrid Attention', 'Hybrid Autoencoder']:
            dummy = tf.random.normal([1, 96, 96, 3])
        else:
            dummy = tf.random.normal([1, 224, 224, 3])
        
        if name == 'ResNet50V2':
            outputs = model(dummy)
            print(f"Output: {outputs}")
        else:
            pred = model.predict(dummy, verbose=0)
            print(f"Prediction shape: {pred.shape}, sum: {pred.sum():.4f}")

except Exception as e:
    print("Error:", e)
    print("Traceback:\n", traceback.format_exc())
